// Copyright 2020 ETH Zurich and University of Bologna.
//
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Author: Matheus Cavalcante, ETH Zurich
//         Samuel Riedel, ETH Zurich
// Pure FP16 refactor: Stavros Mitropoulos, NTUA

#include <stddef.h>
#include <stdint.h>
#include "fmatmul.h"

#define MIN(a, b) ((a) < (b) ? (a) : (b))


// ===========================================================================
// PORTED FROM fc_layer32/kernel/fmatmul.c (outer-product / broadcast form).
//
// The previous FP16 implementation used a dot-product form that was wrong in
// four independent ways:
//   1. all four accumulators received the IDENTICAL product (vfmacc.vv with the
//      same operand pair), so every output row held the same value;
//   2. only row 0 of the A block was ever loaded -- rows 1..3 were never read;
//   3. the p loop advanced by block_size while the body wrote a single element
//      per row, leaving 3 of every 4 output columns untouched;
//   4. the reduction ran at vl = p_ (<= tile size) while the accumulators held
//      up to VLMAX meaningful lanes, so it summed a handful of the products and
//      discarded the rest.
// Net effect: ~1/4 of the multiply-accumulates of the correct kernel, which is
// why the FP16 backward appeared ~4x faster than FP32 -- a ratio no amount of
// precision reduction can produce (the strip-mine bound is ~2x).
//
// This file is now a mechanical e32->e16 translation of the FP32 kernels, so
// the two apps run the SAME algorithm and the comparison is meaningful.
// ===========================================================================


void fmatmul_vec_4x4_slice_init_tn_32() {
  asm volatile("vmv.v.i v0,  0");
  asm volatile("vmv.v.i v4,  0");
  asm volatile("vmv.v.i v8,  0");
  asm volatile("vmv.v.i v12, 0");
}

void fmatmul_vec_8x8_slice_init_tn_32() {
  asm volatile("vmv.v.i v0,  0");
  asm volatile("vmv.v.i v2,  0");
  asm volatile("vmv.v.i v4,  0");
  asm volatile("vmv.v.i v6,  0");
  asm volatile("vmv.v.i v8,  0");
  asm volatile("vmv.v.i v10, 0");
  asm volatile("vmv.v.i v12, 0");
  asm volatile("vmv.v.i v14, 0");
}

void fmatmul_vec_16x16_slice_init_tn_32() {
  asm volatile("vmv.v.i v0,  0");
  asm volatile("vmv.v.i v1,  0");
  asm volatile("vmv.v.i v2,  0");
  asm volatile("vmv.v.i v3,  0");
  asm volatile("vmv.v.i v4,  0");
  asm volatile("vmv.v.i v5,  0");
  asm volatile("vmv.v.i v6,  0");
  asm volatile("vmv.v.i v7,  0");
  asm volatile("vmv.v.i v8,  0");
  asm volatile("vmv.v.i v9,  0");
  asm volatile("vmv.v.i v10, 0");
  asm volatile("vmv.v.i v11, 0");
  asm volatile("vmv.v.i v12, 0");
  asm volatile("vmv.v.i v13, 0");
  asm volatile("vmv.v.i v14, 0");
  asm volatile("vmv.v.i v15, 0");
}

void fmatmul_tn_32(_Float16 *c, const _Float16 *a, const _Float16 *b,
                const unsigned long int M, const unsigned long int N,
                const unsigned long int P) {
  if (N <= 4) {
    fmatmul_4x4_tn_32(c, a, b, M, N, P);
  } else if (N <= 8) {
    fmatmul_8x8_tn_32(c, a, b, M, N, P);
  } else if (N <= 64) {
    fmatmul_16x16_tn_32(c, a, b, M, N, P);
  } else if (N <= 128) {
    // With an 8x8 matmul, use LMUL=2 to increase vl.
    fmatmul_8x8_tn_32(c, a, b, M, N, P);
  } else {
    // With a 4x4 matmul, use LMUL=4 to increase vl.
    fmatmul_4x4_tn_32(c, a, b, M, N, P);
  }

  // NOTE: pure c = a^T * b, NO 1/M scaling. vggnet16 folds the 1/batchsize
  // into the loss seed once (with the FP16 loss scale), matching vggnet's
  // loss-side convention -- see kernel/train.c. fc_layer16only keeps its own
  // copy of this file with the 1/M in place; this one is deliberately without.
}

void fmatmul_4x4_tn_32(_Float16 *c, const _Float16 *a, const _Float16 *b,
                    const unsigned long int M, const unsigned long int N,
                    const unsigned long int P) {
  const unsigned long int block_size = 4;
  unsigned long int block_size_p;

  asm volatile("vsetvli %0, %1, e16, m4, ta, ma" : "=r"(block_size_p) : "r"(P));

  for (unsigned long int p = 0; p < P; p += block_size_p) {
    const unsigned long int p_ = MIN(P - p, block_size_p);

    const _Float16 *b_ = b + p;
    _Float16 *c_ = c + p;

    asm volatile("vsetvli zero, %0, e16, m4, ta, ma" : : "r"(p_));

    for (unsigned long int n = 0; n < N; n += block_size) {
      const _Float16 *a_ = a + n;
      _Float16 *c__ = c_ + n * P;

      fmatmul_vec_4x4_slice_init_tn_32();
      fmatmul_vec_4x4_tn_32(c__, a_, b_, M, P, N);
    }
  }
}

void fmatmul_vec_4x4_tn_32(_Float16 *c, const _Float16 *a, const _Float16 *b,
                        const unsigned long int N, const unsigned long int P,
                        const unsigned long int lda) { //could use smaller LMUL=1/2
  _Float16 t0, t1, t2, t3;

  if (N <= 2) { //batchsize=2
    const _Float16 *a_row = a;
    for (unsigned long int m = 0; m < N; m++) {
      asm volatile("vle16.v v16, (%0);" ::"r"(b));
      b += P;

      t0 = a_row[0];
      t1 = a_row[1];
      t2 = a_row[2];
      t3 = a_row[3];

      asm volatile("vfmacc.vf v0, %0, v16" ::"f"(t0));
      asm volatile("vfmacc.vf v4, %0, v16" ::"f"(t1));
      asm volatile("vfmacc.vf v8, %0, v16" ::"f"(t2));
      asm volatile("vfmacc.vf v12, %0, v16" ::"f"(t3));

      a_row += lda;
    }

    asm volatile("vse16.v v0, (%0);" ::"r"(c));
    c += P;
    asm volatile("vse16.v v4, (%0);" ::"r"(c));
    c += P;
    asm volatile("vse16.v v8, (%0);" ::"r"(c));
    c += P;
    asm volatile("vse16.v v12, (%0);" ::"r"(c));
    return;
  }

  // Prefetch one row of matrix B.
  asm volatile("vle16.v v16, (%0);" ::"r"(b));
  b += P;

  // TN: load 4 contiguous elements from A, then advance by lda (row stride).
  t0 = a[0];
  t1 = a[1];
  t2 = a[2];
  t3 = a[3];
  a += lda;

  unsigned long int n = 0;

  while (n != N) {
    n++;

    asm volatile("vfmacc.vf v0, %0, v16" ::"f"(t0));
    t0 = a[0];

    asm volatile("vle16.v v20, (%0);" ::"r"(b));
    b += P;

    asm volatile("vfmacc.vf v4, %0, v16" ::"f"(t1));
    t1 = a[1];
    asm volatile("vfmacc.vf v8, %0, v16" ::"f"(t2));
    t2 = a[2];
    asm volatile("vfmacc.vf v12, %0, v16" ::"f"(t3));
    t3 = a[3];

    a += lda;

    if (n == N)
      break;

    n++;

    asm volatile("vfmacc.vf v0, %0, v20" ::"f"(t0));
    t0 = a[0];

    asm volatile("vle16.v v16, (%0);" ::"r"(b));
    b += P;

    asm volatile("vfmacc.vf v4, %0, v20" ::"f"(t1));
    t1 = a[1];
    asm volatile("vfmacc.vf v8, %0, v20" ::"f"(t2));
    t2 = a[2];
    asm volatile("vfmacc.vf v12, %0, v20" ::"f"(t3));
    t3 = a[3];

    a += lda;
  }

  asm volatile("vfmacc.vf v0, %0, v20" ::"f"(t0));
  asm volatile("vse16.v v0, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfmacc.vf v4, %0, v20" ::"f"(t1));
  asm volatile("vse16.v v4, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfmacc.vf v8, %0, v20" ::"f"(t2));
  asm volatile("vse16.v v8, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfmacc.vf v12, %0, v20" ::"f"(t3));
  asm volatile("vse16.v v12, (%0);" ::"r"(c));
}

void fmatmul_8x8_tn_32(_Float16 *c, const _Float16 *a, const _Float16 *b,
                    const unsigned long int M, const unsigned long int N,
                    const unsigned long int P) {
  const unsigned long int block_size = 8;
  unsigned long int block_size_p;

  asm volatile("vsetvli %0, %1, e16, m2, ta, ma" : "=r"(block_size_p) : "r"(P));

  for (unsigned long int p = 0; p < P; p += block_size_p) {
    const unsigned long int p_ = MIN(P - p, block_size_p);

    const _Float16 *b_ = b + p;
    _Float16 *c_ = c + p;

    asm volatile("vsetvli zero, %0, e16, m2, ta, ma" : : "r"(p_));

    for (unsigned long int n = 0; n < N; n += block_size) {
      const _Float16 *a_ = a + n;
      _Float16 *c__ = c_ + n * P;

      fmatmul_vec_8x8_slice_init_tn_32();
      fmatmul_vec_8x8_tn_32(c__, a_, b_, M, P, N);
    }
  }
}

void fmatmul_vec_8x8_tn_32(_Float16 *c, const _Float16 *a, const _Float16 *b,
                        const unsigned long int N, const unsigned long int P,
                        const unsigned long int lda) {
  _Float16 t0, t1, t2, t3, t4, t5, t6, t7;

  if (N <= 2) {
    const _Float16 *a_row = a;
    for (unsigned long int m = 0; m < N; m++) {
      asm volatile("vle16.v v18, (%0);" ::"r"(b));
      b += P;

      t0 = a_row[0];
      t1 = a_row[1];
      t2 = a_row[2];
      t3 = a_row[3];
      t4 = a_row[4];
      t5 = a_row[5];
      t6 = a_row[6];
      t7 = a_row[7];

      asm volatile("vfmacc.vf v0, %0, v18" ::"f"(t0));
      asm volatile("vfmacc.vf v2, %0, v18" ::"f"(t1));
      asm volatile("vfmacc.vf v4, %0, v18" ::"f"(t2));
      asm volatile("vfmacc.vf v6, %0, v18" ::"f"(t3));
      asm volatile("vfmacc.vf v8, %0, v18" ::"f"(t4));
      asm volatile("vfmacc.vf v10, %0, v18" ::"f"(t5));
      asm volatile("vfmacc.vf v12, %0, v18" ::"f"(t6));
      asm volatile("vfmacc.vf v14, %0, v18" ::"f"(t7));

      a_row += lda;
    }

    asm volatile("vse16.v v0, (%0);" ::"r"(c));
    c += P;
    asm volatile("vse16.v v2, (%0);" ::"r"(c));
    c += P;
    asm volatile("vse16.v v4, (%0);" ::"r"(c));
    c += P;
    asm volatile("vse16.v v6, (%0);" ::"r"(c));
    c += P;
    asm volatile("vse16.v v8, (%0);" ::"r"(c));
    c += P;
    asm volatile("vse16.v v10, (%0);" ::"r"(c));
    c += P;
    asm volatile("vse16.v v12, (%0);" ::"r"(c));
    c += P;
    asm volatile("vse16.v v14, (%0);" ::"r"(c));
    return;
  }

  asm volatile("vle16.v v18, (%0);" ::"r"(b));
  b += P;

  t0 = a[0];
  t1 = a[1];
  t2 = a[2];
  t3 = a[3];
  t4 = a[4];
  t5 = a[5];
  t6 = a[6];
  t7 = a[7];
  a += lda;

  unsigned long int n = 0;

  while (n != N) {
#ifdef VCD_DUMP
    if (n == 8)
      event_trigger = +1;
    if (n == 12)
      event_trigger = -1;
#endif

    n++;

    asm volatile("vfmacc.vf v0, %0, v18" ::"f"(t0));
    t0 = a[0];

    asm volatile("vle16.v v20, (%0);" ::"r"(b));
    b += P;

    asm volatile("vfmacc.vf v2, %0, v18" ::"f"(t1));
    t1 = a[1];
    asm volatile("vfmacc.vf v4, %0, v18" ::"f"(t2));
    t2 = a[2];
    asm volatile("vfmacc.vf v6, %0, v18" ::"f"(t3));
    t3 = a[3];
    asm volatile("vfmacc.vf v8, %0, v18" ::"f"(t4));
    t4 = a[4];
    asm volatile("vfmacc.vf v10, %0, v18" ::"f"(t5));
    t5 = a[5];
    asm volatile("vfmacc.vf v12, %0, v18" ::"f"(t6));
    t6 = a[6];
    asm volatile("vfmacc.vf v14, %0, v18" ::"f"(t7));
    t7 = a[7];

    a += lda;

    if (n == N)
      break;

    n++;

    asm volatile("vfmacc.vf v0, %0, v20" ::"f"(t0));
    t0 = a[0];

    asm volatile("vle16.v v18, (%0);" ::"r"(b));
    b += P;

    asm volatile("vfmacc.vf v2, %0, v20" ::"f"(t1));
    t1 = a[1];
    asm volatile("vfmacc.vf v4, %0, v20" ::"f"(t2));
    t2 = a[2];
    asm volatile("vfmacc.vf v6, %0, v20" ::"f"(t3));
    t3 = a[3];
    asm volatile("vfmacc.vf v8, %0, v20" ::"f"(t4));
    t4 = a[4];
    asm volatile("vfmacc.vf v10, %0, v20" ::"f"(t5));
    t5 = a[5];
    asm volatile("vfmacc.vf v12, %0, v20" ::"f"(t6));
    t6 = a[6];
    asm volatile("vfmacc.vf v14, %0, v20" ::"f"(t7));
    t7 = a[7];

    a += lda;
  }

  asm volatile("vfmacc.vf v0, %0, v20" ::"f"(t0));
  asm volatile("vse16.v v0, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfmacc.vf v2, %0, v20" ::"f"(t1));
  asm volatile("vse16.v v2, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfmacc.vf v4, %0, v20" ::"f"(t2));
  asm volatile("vse16.v v4, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfmacc.vf v6, %0, v20" ::"f"(t3));
  asm volatile("vse16.v v6, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfmacc.vf v8, %0, v20" ::"f"(t4));
  asm volatile("vse16.v v8, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfmacc.vf v10, %0, v20" ::"f"(t5));
  asm volatile("vse16.v v10, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfmacc.vf v12, %0, v20" ::"f"(t6));
  asm volatile("vse16.v v12, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfmacc.vf v14, %0, v20" ::"f"(t7));
  asm volatile("vse16.v v14, (%0);" ::"r"(c));
}

void fmatmul_16x16_tn_32(_Float16 *c, const _Float16 *a, const _Float16 *b,
                      unsigned long int M, unsigned long int N,
                      unsigned long int P) {
  const unsigned long int block_size = 16;
  unsigned long int block_size_p;

  asm volatile("vsetvli %0, %1, e16, m1, ta, ma" : "=r"(block_size_p) : "r"(P));

  for (unsigned long int p = 0; p < P; p += block_size_p) {
    const unsigned long int p_ = MIN(P - p, block_size_p);

    const _Float16 *b_ = b + p;
    _Float16 *c_ = c + p;

    asm volatile("vsetvli zero, %0, e16, m1, ta, ma" : : "r"(p_));

    for (unsigned long int n = 0; n < N; n += block_size) {
      const _Float16 *a_ = a + n;
      _Float16 *c__ = c_ + n * P;

      fmatmul_vec_16x16_slice_init_tn_32();
      fmatmul_vec_16x16_tn_32(c__, a_, b_, M, P, N);
    }
  }
}

void fmatmul_vec_16x16_tn_32(_Float16 *c, const _Float16 *a, const _Float16 *b,
                          const unsigned long int N, const unsigned long int P,
                          const unsigned long int lda) {
  _Float16 t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15;

  if (N <= 2) {
    const _Float16 *a_row = a;
    for (unsigned long int m = 0; m < N; m++) {
      asm volatile("vle16.v v16, (%0);" ::"r"(b));
      b += P;

      t0 = a_row[0];
      t1 = a_row[1];
      t2 = a_row[2];
      t3 = a_row[3];
      t4 = a_row[4];
      t5 = a_row[5];
      t6 = a_row[6];
      t7 = a_row[7];
      t8 = a_row[8];
      t9 = a_row[9];
      t10 = a_row[10];
      t11 = a_row[11];
      t12 = a_row[12];
      t13 = a_row[13];
      t14 = a_row[14];
      t15 = a_row[15];

      asm volatile("vfmacc.vf v0, %0, v16" ::"f"(t0));
      asm volatile("vfmacc.vf v1, %0, v16" ::"f"(t1));
      asm volatile("vfmacc.vf v2, %0, v16" ::"f"(t2));
      asm volatile("vfmacc.vf v3, %0, v16" ::"f"(t3));
      asm volatile("vfmacc.vf v4, %0, v16" ::"f"(t4));
      asm volatile("vfmacc.vf v5, %0, v16" ::"f"(t5));
      asm volatile("vfmacc.vf v6, %0, v16" ::"f"(t6));
      asm volatile("vfmacc.vf v7, %0, v16" ::"f"(t7));
      asm volatile("vfmacc.vf v8, %0, v16" ::"f"(t8));
      asm volatile("vfmacc.vf v9, %0, v16" ::"f"(t9));
      asm volatile("vfmacc.vf v10, %0, v16" ::"f"(t10));
      asm volatile("vfmacc.vf v11, %0, v16" ::"f"(t11));
      asm volatile("vfmacc.vf v12, %0, v16" ::"f"(t12));
      asm volatile("vfmacc.vf v13, %0, v16" ::"f"(t13));
      asm volatile("vfmacc.vf v14, %0, v16" ::"f"(t14));
      asm volatile("vfmacc.vf v15, %0, v16" ::"f"(t15));

      a_row += lda;
    }

    asm volatile("vse16.v v0, (%0);" ::"r"(c));
    c += P;
    asm volatile("vse16.v v1, (%0);" ::"r"(c));
    c += P;
    asm volatile("vse16.v v2, (%0);" ::"r"(c));
    c += P;
    asm volatile("vse16.v v3, (%0);" ::"r"(c));
    c += P;
    asm volatile("vse16.v v4, (%0);" ::"r"(c));
    c += P;
    asm volatile("vse16.v v5, (%0);" ::"r"(c));
    c += P;
    asm volatile("vse16.v v6, (%0);" ::"r"(c));
    c += P;
    asm volatile("vse16.v v7, (%0);" ::"r"(c));
    c += P;
    asm volatile("vse16.v v8, (%0);" ::"r"(c));
    c += P;
    asm volatile("vse16.v v9, (%0);" ::"r"(c));
    c += P;
    asm volatile("vse16.v v10, (%0);" ::"r"(c));
    c += P;
    asm volatile("vse16.v v11, (%0);" ::"r"(c));
    c += P;
    asm volatile("vse16.v v12, (%0);" ::"r"(c));
    c += P;
    asm volatile("vse16.v v13, (%0);" ::"r"(c));
    c += P;
    asm volatile("vse16.v v14, (%0);" ::"r"(c));
    c += P;
    asm volatile("vse16.v v15, (%0);" ::"r"(c));
    return;
  }

  t0 = a[0];
  t1 = a[1];
  t2 = a[2];
  t3 = a[3];
  t4 = a[4];
  t5 = a[5];
  t6 = a[6];
  t7 = a[7];
  t8 = a[8];
  t9 = a[9];
  t10 = a[10];
  t11 = a[11];
  t12 = a[12];
  t13 = a[13];
  t14 = a[14];
  t15 = a[15];

  a += lda;

  asm volatile("vle16.v v16, (%0);" ::"r"(b));
  b += P;

  unsigned long int n = 0;

  while (n != N) {
#ifdef VCD_DUMP
    if (n == 8)
      event_trigger = +1;
    if (n == 12)
      event_trigger = -1;
#endif

    n++;

    asm volatile("vfmacc.vf v0, %0, v16" ::"f"(t0));
    t0 = a[0];

    asm volatile("vle16.v v17, (%0);" ::"r"(b));
    b += P;

    asm volatile("vfmacc.vf v1, %0, v16" ::"f"(t1));
    t1 = a[1];
    asm volatile("vfmacc.vf v2, %0, v16" ::"f"(t2));
    t2 = a[2];
    asm volatile("vfmacc.vf v3, %0, v16" ::"f"(t3));
    t3 = a[3];
    asm volatile("vfmacc.vf v4, %0, v16" ::"f"(t4));
    t4 = a[4];
    asm volatile("vfmacc.vf v5, %0, v16" ::"f"(t5));
    t5 = a[5];
    asm volatile("vfmacc.vf v6, %0, v16" ::"f"(t6));
    t6 = a[6];
    asm volatile("vfmacc.vf v7, %0, v16" ::"f"(t7));
    t7 = a[7];
    asm volatile("vfmacc.vf v8, %0, v16" ::"f"(t8));
    t8 = a[8];
    asm volatile("vfmacc.vf v9, %0, v16" ::"f"(t9));
    t9 = a[9];
    asm volatile("vfmacc.vf v10, %0, v16" ::"f"(t10));
    t10 = a[10];
    asm volatile("vfmacc.vf v11, %0, v16" ::"f"(t11));
    t11 = a[11];
    asm volatile("vfmacc.vf v12, %0, v16" ::"f"(t12));
    t12 = a[12];
    asm volatile("vfmacc.vf v13, %0, v16" ::"f"(t13));
    t13 = a[13];
    asm volatile("vfmacc.vf v14, %0, v16" ::"f"(t14));
    t14 = a[14];
    asm volatile("vfmacc.vf v15, %0, v16" ::"f"(t15));
    t15 = a[15];

    a += lda;

    if (n == N)
      break;

    n++;

    asm volatile("vfmacc.vf v0, %0, v17" ::"f"(t0));
    t0 = a[0];

    asm volatile("vle16.v v16, (%0);" ::"r"(b));
    b += P;

    asm volatile("vfmacc.vf v1, %0, v17" ::"f"(t1));
    t1 = a[1];
    asm volatile("vfmacc.vf v2, %0, v17" ::"f"(t2));
    t2 = a[2];
    asm volatile("vfmacc.vf v3, %0, v17" ::"f"(t3));
    t3 = a[3];
    asm volatile("vfmacc.vf v4, %0, v17" ::"f"(t4));
    t4 = a[4];
    asm volatile("vfmacc.vf v5, %0, v17" ::"f"(t5));
    t5 = a[5];
    asm volatile("vfmacc.vf v6, %0, v17" ::"f"(t6));
    t6 = a[6];
    asm volatile("vfmacc.vf v7, %0, v17" ::"f"(t7));
    t7 = a[7];
    asm volatile("vfmacc.vf v8, %0, v17" ::"f"(t8));
    t8 = a[8];
    asm volatile("vfmacc.vf v9, %0, v17" ::"f"(t9));
    t9 = a[9];
    asm volatile("vfmacc.vf v10, %0, v17" ::"f"(t10));
    t10 = a[10];
    asm volatile("vfmacc.vf v11, %0, v17" ::"f"(t11));
    t11 = a[11];
    asm volatile("vfmacc.vf v12, %0, v17" ::"f"(t12));
    t12 = a[12];
    asm volatile("vfmacc.vf v13, %0, v17" ::"f"(t13));
    t13 = a[13];
    asm volatile("vfmacc.vf v14, %0, v17" ::"f"(t14));
    t14 = a[14];
    asm volatile("vfmacc.vf v15, %0, v17" ::"f"(t15));
    t15 = a[15];

    a += lda;
  }

  asm volatile("vfmacc.vf v0, %0, v17" ::"f"(t0));
  asm volatile("vse16.v v0, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfmacc.vf v1, %0, v17" ::"f"(t1));
  asm volatile("vse16.v v1, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfmacc.vf v2, %0, v17" ::"f"(t2));
  asm volatile("vse16.v v2, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfmacc.vf v3, %0, v17" ::"f"(t3));
  asm volatile("vse16.v v3, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfmacc.vf v4, %0, v17" ::"f"(t4));
  asm volatile("vse16.v v4, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfmacc.vf v5, %0, v17" ::"f"(t5));
  asm volatile("vse16.v v5, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfmacc.vf v6, %0, v17" ::"f"(t6));
  asm volatile("vse16.v v6, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfmacc.vf v7, %0, v17" ::"f"(t7));
  asm volatile("vse16.v v7, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfmacc.vf v8, %0, v17" ::"f"(t8));
  asm volatile("vse16.v v8, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfmacc.vf v9, %0, v17" ::"f"(t9));
  asm volatile("vse16.v v9, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfmacc.vf v10, %0, v17" ::"f"(t10));
  asm volatile("vse16.v v10, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfmacc.vf v11, %0, v17" ::"f"(t11));
  asm volatile("vse16.v v11, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfmacc.vf v12, %0, v17" ::"f"(t12));
  asm volatile("vse16.v v12, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfmacc.vf v13, %0, v17" ::"f"(t13));
  asm volatile("vse16.v v13, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfmacc.vf v14, %0, v17" ::"f"(t14));
  asm volatile("vse16.v v14, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfmacc.vf v15, %0, v17" ::"f"(t15));
  asm volatile("vse16.v v15, (%0);" ::"r"(c));
}
