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
// modified versions of fmatmul Author: Stavros Mitropoulos, NTUA

#include <stddef.h>
#include <stdint.h>
#include "fmatmul.h"

#define MIN(a, b) ((a) < (b) ? (a) : (b))

void fmatmul_tn(float *c, const _Float16 *a, const _Float16 *b,
                const unsigned long int M, const unsigned long int N,
                const unsigned long int P) {
  if (N <= 4) {
    fmatmul_4x4_tn(c, a, b, M, N, P);
  } else if (N <= 8) {
    fmatmul_8x8_tn(c, a, b, M, N, P);
  } else if (N <= 64) {
    fmatmul_16x16_tn(c, a, b, M, N, P);
  } else if (N <= 128) {
    // With an 8x8 matmul, use LMUL=2 to increase vl.
    fmatmul_8x8_tn(c, a, b, M, N, P);
  } else {
    // With a 4x4 matmul, use LMUL=4 to increase vl.
    fmatmul_4x4_tn(c, a, b, M, N, P);
  }

  if (M == 0)
    return;

  // Scale by 1/M using vector instructions. c is FP32 output.
  // (uint32_t) narrowing is REQUIRED on this hardware, not cosmetic: (float)M on a
  // 64-bit M emits fcvt.s.lu, which does not read its operand (RVD=0 => scalar
  // FLen=32 => fpnew masks INT64 out of IntFmtConfig, and the cast unit returns
  // the inactive format's all-ones DONT_CARE). It yields 2^32 for ANY input,
  // silently zeroing this whole scale pass. fcvt.s.wu (INT32) is correct here.
  // Matrix dimensions are far below 2^32, so the narrowing is lossless.
  // See AGENT_NOTES_FP16_4LANE.md 43.3 / 46.8 / 47.
  const float inv_batch = 1.0f / (float)(uint32_t)M;
  size_t remaining = (size_t)N * (size_t)P;
  float *dst = c;
  while (remaining > 0) {
    size_t vl = 0;
    asm volatile("vsetvli %0, %1, e32, m8, ta, ma" : "=r"(vl) : "r"(remaining));
    asm volatile("vle32.v v8, (%0)" :: "r"(dst) : "memory");
    asm volatile("vfmul.vf v8, v8, %0" :: "f"(inv_batch));
    asm volatile("vse32.v v8, (%0)" :: "r"(dst) : "memory");
    dst += vl;
    remaining -= vl;
  }
}

// ---------------
// 4x4
// ---------------

void fmatmul_4x4_tn(float *c, const _Float16 *a, const _Float16 *b,
                    const unsigned long int M, const unsigned long int N,
                    const unsigned long int P) {
  const unsigned long int block_size = 4;
  unsigned long int block_size_p;

  asm volatile("vsetvli %0, %1, e16, m2, ta, ma" : "=r"(block_size_p) : "r"(P));

  for (unsigned long int p = 0; p < P; p += block_size_p) {
    const unsigned long int p_ = MIN(P - p, block_size_p);

    const _Float16 *b_ = b + p;
    float *c_ = c + p;

    asm volatile("vsetvli zero, %0, e16, m2, ta, ma" : : "r"(p_));

    for (unsigned long int n = 0; n < N; n += block_size) {
      const _Float16 *a_ = a + n;
      float *c__ = c_ + n * P;

      fmatmul_vec_4x4_slice_init(p_);
      fmatmul_vec_4x4_tn(c__, a_, b_, M, P, N, p_);
    }
  }
}

// ---------------
// 4x4 deferred (A * B^T)
// ---------------
void fmatmul_vec_4x4_tn(float *c, const _Float16 *a, const _Float16 *b,
                        const unsigned long int N, const unsigned long int P,
                        const unsigned long int lda, const unsigned long int vl) {
  _Float16 t0, t1, t2, t3;

  asm volatile("vsetvli zero, %0, e16, m2, ta, ma" :: "r"(vl));
  if (N <= 2) { //batchsize=2
    const _Float16 *a_row = a;
    for (unsigned long int m = 0; m < N; m++) {
      asm volatile("vle16.v v16, (%0);" ::"r"(b));
      b += P;

      t0 = a_row[0];
      t1 = a_row[1];
      t2 = a_row[2];
      t3 = a_row[3];

      asm volatile("vfwmacc.vf v0, %0, v16" ::"f"(t0));
      asm volatile("vfwmacc.vf v4, %0, v16" ::"f"(t1));
      asm volatile("vfwmacc.vf v8, %0, v16" ::"f"(t2));
      asm volatile("vfwmacc.vf v12, %0, v16" ::"f"(t3));

      a_row += lda;
    }

    asm volatile("vse32.v v0, (%0);" ::"r"(c));
    c += P;
    asm volatile("vse32.v v4, (%0);" ::"r"(c));
    c += P;
    asm volatile("vse32.v v8, (%0);" ::"r"(c));
    c += P;
    asm volatile("vse32.v v12, (%0);" ::"r"(c));
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

    asm volatile("vfwmacc.vf v0, %0, v16" ::"f"(t0));
    t0 = a[0];

    asm volatile("vle16.v v20, (%0);" ::"r"(b));
    b += P;

    asm volatile("vfwmacc.vf v4, %0, v16" ::"f"(t1));
    t1 = a[1];
    asm volatile("vfwmacc.vf v8, %0, v16" ::"f"(t2));
    t2 = a[2];
    asm volatile("vfwmacc.vf v12, %0, v16" ::"f"(t3));
    t3 = a[3];

    a += lda;

    if (n == N)
      break;

    n++;

    asm volatile("vfwmacc.vf v0, %0, v20" ::"f"(t0));
    t0 = a[0];

    asm volatile("vle16.v v16, (%0);" ::"r"(b));
    b += P;

    asm volatile("vfwmacc.vf v4, %0, v20" ::"f"(t1));
    t1 = a[1];
    asm volatile("vfwmacc.vf v8, %0, v20" ::"f"(t2));
    t2 = a[2];
    asm volatile("vfwmacc.vf v12, %0, v20" ::"f"(t3));
    t3 = a[3];

    a += lda;
  }

  asm volatile("vfwmacc.vf v0, %0, v20" ::"f"(t0));
  asm volatile("vfwmacc.vf v4, %0, v20" ::"f"(t1));
  asm volatile("vfwmacc.vf v8, %0, v20" ::"f"(t2));
  asm volatile("vfwmacc.vf v12, %0, v20" ::"f"(t3));

  asm volatile("vsetvli zero, %0, e32, m4, ta, ma" :: "r"(vl));
  asm volatile("vse32.v v0, (%0);" ::"r"(c));
  c += P;
  
  asm volatile("vse32.v v4, (%0);" ::"r"(c));
  c += P;
  
  asm volatile("vse32.v v8, (%0);" ::"r"(c));
  c += P;
  asm volatile("vse32.v v12, (%0);" ::"r"(c));
}

// ---------------
// 8x8
// ---------------

void fmatmul_8x8_tn(float *c, const _Float16 *a, const _Float16 *b,
                    const unsigned long int M, const unsigned long int N,
                    const unsigned long int P) {
  const unsigned long int block_size = 8;
  unsigned long int block_size_p;

  asm volatile("vsetvli %0, %1, e16, m1, ta, ma" : "=r"(block_size_p) : "r"(P));

  for (unsigned long int p = 0; p < P; p += block_size_p) {
    const unsigned long int p_ = MIN(P - p, block_size_p);

    const _Float16 *b_ = b + p;
    float *c_ = c + p;

    asm volatile("vsetvli zero, %0, e16, m1, ta, ma" : : "r"(p_));

    for (unsigned long int n = 0; n < N; n += block_size) {
      const _Float16 *a_ = a + n;
      float *c__ = c_ + n * P;

      fmatmul_vec_8x8_slice_init(p_);
      fmatmul_vec_8x8_tn(c__, a_, b_, M, P, N, p_);
    }
  }
}

void fmatmul_vec_8x8_tn(float *c, const _Float16 *a, const _Float16 *b,
                        const unsigned long int N, const unsigned long int P,
                        const unsigned long int lda, const unsigned long int vl) {
  _Float16 t0, t1, t2, t3, t4, t5, t6, t7;

  asm volatile("vsetvli zero, %0, e16, m1, ta, ma" :: "r"(vl));
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

      asm volatile("vfwmacc.vf v0, %0, v18" ::"f"(t0));
      asm volatile("vfwmacc.vf v2, %0, v18" ::"f"(t1));
      asm volatile("vfwmacc.vf v4, %0, v18" ::"f"(t2));
      asm volatile("vfwmacc.vf v6, %0, v18" ::"f"(t3));
      asm volatile("vfwmacc.vf v8, %0, v18" ::"f"(t4));
      asm volatile("vfwmacc.vf v10, %0, v18" ::"f"(t5));
      asm volatile("vfwmacc.vf v12, %0, v18" ::"f"(t6));
      asm volatile("vfwmacc.vf v14, %0, v18" ::"f"(t7));

      a_row += lda;
    }

    asm volatile("vsetvli zero, %0, e32, m2, ta, ma" :: "r"(vl));
    asm volatile("vse32.v v0, (%0);" ::"r"(c));
    c += P;
    asm volatile("vse32.v v2, (%0);" ::"r"(c));
    c += P;
    asm volatile("vse32.v v4, (%0);" ::"r"(c));
    c += P;
    asm volatile("vse32.v v6, (%0);" ::"r"(c));
    c += P;
    asm volatile("vse32.v v8, (%0);" ::"r"(c));
    c += P;
    asm volatile("vse32.v v10, (%0);" ::"r"(c));
    c += P;
    asm volatile("vse32.v v12, (%0);" ::"r"(c));
    c += P;
    asm volatile("vse32.v v14, (%0);" ::"r"(c));
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

    asm volatile("vfwmacc.vf v0, %0, v18" ::"f"(t0));
    t0 = a[0];

    asm volatile("vle16.v v20, (%0);" ::"r"(b));
    b += P;

    asm volatile("vfwmacc.vf v2, %0, v18" ::"f"(t1));
    t1 = a[1];
    asm volatile("vfwmacc.vf v4, %0, v18" ::"f"(t2));
    t2 = a[2];
    asm volatile("vfwmacc.vf v6, %0, v18" ::"f"(t3));
    t3 = a[3];
    asm volatile("vfwmacc.vf v8, %0, v18" ::"f"(t4));
    t4 = a[4];
    asm volatile("vfwmacc.vf v10, %0, v18" ::"f"(t5));
    t5 = a[5];
    asm volatile("vfwmacc.vf v12, %0, v18" ::"f"(t6));
    t6 = a[6];
    asm volatile("vfwmacc.vf v14, %0, v18" ::"f"(t7));
    t7 = a[7];

    a += lda;

    if (n == N)
      break;

    n++;

    asm volatile("vfwmacc.vf v0, %0, v20" ::"f"(t0));
    t0 = a[0];

    asm volatile("vle16.v v18, (%0);" ::"r"(b));
    b += P;

    asm volatile("vfwmacc.vf v2, %0, v20" ::"f"(t1));
    t1 = a[1];
    asm volatile("vfwmacc.vf v4, %0, v20" ::"f"(t2));
    t2 = a[2];
    asm volatile("vfwmacc.vf v6, %0, v20" ::"f"(t3));
    t3 = a[3];
    asm volatile("vfwmacc.vf v8, %0, v20" ::"f"(t4));
    t4 = a[4];
    asm volatile("vfwmacc.vf v10, %0, v20" ::"f"(t5));
    t5 = a[5];
    asm volatile("vfwmacc.vf v12, %0, v20" ::"f"(t6));
    t6 = a[6];
    asm volatile("vfwmacc.vf v14, %0, v20" ::"f"(t7));
    t7 = a[7];

    a += lda;
  }

// Last iteration: final math (Ακόμα σε e16, m1)
  asm volatile("vfwmacc.vf v0, %0, v20" ::"f"(t0));
  asm volatile("vfwmacc.vf v2, %0, v20" ::"f"(t1));
  asm volatile("vfwmacc.vf v4, %0, v20" ::"f"(t2));
  asm volatile("vfwmacc.vf v6, %0, v20" ::"f"(t3));
  asm volatile("vfwmacc.vf v8, %0, v20" ::"f"(t4));
  asm volatile("vfwmacc.vf v10, %0, v20" ::"f"(t5));
  asm volatile("vfwmacc.vf v12, %0, v20" ::"f"(t6));
  asm volatile("vfwmacc.vf v14, %0, v20" ::"f"(t7));

  // --- ΚΡΙΣΙΜΟ CONTEXT SWITCH ---
  asm volatile("vsetvli zero, %0, e32, m2, ta, ma" :: "r"(vl));

  // Memory stores (e32 mode)
  asm volatile("vse32.v v0, (%0);" ::"r"(c));
  c += P;
  asm volatile("vse32.v v2, (%0);" ::"r"(c));
  c += P;
  asm volatile("vse32.v v4, (%0);" ::"r"(c));
  c += P;
  asm volatile("vse32.v v6, (%0);" ::"r"(c));
  c += P;
  asm volatile("vse32.v v8, (%0);" ::"r"(c));
  c += P;
  asm volatile("vse32.v v10, (%0);" ::"r"(c));
  c += P;
  asm volatile("vse32.v v12, (%0);" ::"r"(c));
  c += P;
  asm volatile("vse32.v v14, (%0);" ::"r"(c));
}

// ---------------
// 16x16
// ---------------

void fmatmul_16x16_tn(float *c, const _Float16 *a, const _Float16 *b,
                      unsigned long int M, unsigned long int N,
                      unsigned long int P) {
  const unsigned long int block_size = 16;
  unsigned long int block_size_p;

  asm volatile("vsetvli %0, %1, e16, mf2, ta, ma" : "=r"(block_size_p) : "r"(P));

  for (unsigned long int p = 0; p < P; p += block_size_p) {
    const unsigned long int p_ = MIN(P - p, block_size_p);

    const _Float16 *b_ = b + p;
    float *c_ = c + p;

    asm volatile("vsetvli zero, %0, e16, mf2, ta, ma" : : "r"(p_));

    for (unsigned long int n = 0; n < N; n += block_size) {
      const _Float16 *a_ = a + n;
      float *c__ = c_ + n * P;

      fmatmul_vec_16x16_slice_init(p_);
      fmatmul_vec_16x16_tn(c__, a_, b_, M, P, N, p_);
    }
  }
}

void fmatmul_vec_16x16_tn(float *c, const _Float16 *a, const _Float16 *b,
                          const unsigned long int N, const unsigned long int P,
                          const unsigned long int lda, const unsigned long int vl) {
  _Float16 t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15;

  asm volatile("vsetvli zero, %0, e16, mf2, ta, ma" :: "r"(vl));
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

      asm volatile("vfwmacc.vf v0, %0, v16" ::"f"(t0));
      asm volatile("vfwmacc.vf v1, %0, v16" ::"f"(t1));
      asm volatile("vfwmacc.vf v2, %0, v16" ::"f"(t2));
      asm volatile("vfwmacc.vf v3, %0, v16" ::"f"(t3));
      asm volatile("vfwmacc.vf v4, %0, v16" ::"f"(t4));
      asm volatile("vfwmacc.vf v5, %0, v16" ::"f"(t5));
      asm volatile("vfwmacc.vf v6, %0, v16" ::"f"(t6));
      asm volatile("vfwmacc.vf v7, %0, v16" ::"f"(t7));
      asm volatile("vfwmacc.vf v8, %0, v16" ::"f"(t8));
      asm volatile("vfwmacc.vf v9, %0, v16" ::"f"(t9));
      asm volatile("vfwmacc.vf v10, %0, v16" ::"f"(t10));
      asm volatile("vfwmacc.vf v11, %0, v16" ::"f"(t11));
      asm volatile("vfwmacc.vf v12, %0, v16" ::"f"(t12));
      asm volatile("vfwmacc.vf v13, %0, v16" ::"f"(t13));
      asm volatile("vfwmacc.vf v14, %0, v16" ::"f"(t14));
      asm volatile("vfwmacc.vf v15, %0, v16" ::"f"(t15));

      a_row += lda;
    }

    asm volatile("vse32.v v0, (%0);" ::"r"(c));
    c += P;
    asm volatile("vse32.v v1, (%0);" ::"r"(c));
    c += P;
    asm volatile("vse32.v v2, (%0);" ::"r"(c));
    c += P;
    asm volatile("vse32.v v3, (%0);" ::"r"(c));
    c += P;
    asm volatile("vse32.v v4, (%0);" ::"r"(c));
    c += P;
    asm volatile("vse32.v v5, (%0);" ::"r"(c));
    c += P;
    asm volatile("vse32.v v6, (%0);" ::"r"(c));
    c += P;
    asm volatile("vse32.v v7, (%0);" ::"r"(c));
    c += P;
    asm volatile("vse32.v v8, (%0);" ::"r"(c));
    c += P;
    asm volatile("vse32.v v9, (%0);" ::"r"(c));
    c += P;
    asm volatile("vse32.v v10, (%0);" ::"r"(c));
    c += P;
    asm volatile("vse32.v v11, (%0);" ::"r"(c));
    c += P;
    asm volatile("vse32.v v12, (%0);" ::"r"(c));
    c += P;
    asm volatile("vse32.v v13, (%0);" ::"r"(c));
    c += P;
    asm volatile("vse32.v v14, (%0);" ::"r"(c));
    c += P;
    asm volatile("vse32.v v15, (%0);" ::"r"(c));
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

    asm volatile("vfwmacc.vf v0, %0, v16" ::"f"(t0));
    t0 = a[0];

    asm volatile("vle16.v v17, (%0);" ::"r"(b));
    b += P;

    asm volatile("vfwmacc.vf v1, %0, v16" ::"f"(t1));
    t1 = a[1];
    asm volatile("vfwmacc.vf v2, %0, v16" ::"f"(t2));
    t2 = a[2];
    asm volatile("vfwmacc.vf v3, %0, v16" ::"f"(t3));
    t3 = a[3];
    asm volatile("vfwmacc.vf v4, %0, v16" ::"f"(t4));
    t4 = a[4];
    asm volatile("vfwmacc.vf v5, %0, v16" ::"f"(t5));
    t5 = a[5];
    asm volatile("vfwmacc.vf v6, %0, v16" ::"f"(t6));
    t6 = a[6];
    asm volatile("vfwmacc.vf v7, %0, v16" ::"f"(t7));
    t7 = a[7];
    asm volatile("vfwmacc.vf v8, %0, v16" ::"f"(t8));
    t8 = a[8];
    asm volatile("vfwmacc.vf v9, %0, v16" ::"f"(t9));
    t9 = a[9];
    asm volatile("vfwmacc.vf v10, %0, v16" ::"f"(t10));
    t10 = a[10];
    asm volatile("vfwmacc.vf v11, %0, v16" ::"f"(t11));
    t11 = a[11];
    asm volatile("vfwmacc.vf v12, %0, v16" ::"f"(t12));
    t12 = a[12];
    asm volatile("vfwmacc.vf v13, %0, v16" ::"f"(t13));
    t13 = a[13];
    asm volatile("vfwmacc.vf v14, %0, v16" ::"f"(t14));
    t14 = a[14];
    asm volatile("vfwmacc.vf v15, %0, v16" ::"f"(t15));
    t15 = a[15];

    a += lda;

    if (n == N)
      break;

    n++;

    asm volatile("vfwmacc.vf v0, %0, v17" ::"f"(t0));
    t0 = a[0];

    asm volatile("vle16.v v16, (%0);" ::"r"(b));
    b += P;

    asm volatile("vfwmacc.vf v1, %0, v17" ::"f"(t1));
    t1 = a[1];
    asm volatile("vfwmacc.vf v2, %0, v17" ::"f"(t2));
    t2 = a[2];
    asm volatile("vfwmacc.vf v3, %0, v17" ::"f"(t3));
    t3 = a[3];
    asm volatile("vfwmacc.vf v4, %0, v17" ::"f"(t4));
    t4 = a[4];
    asm volatile("vfwmacc.vf v5, %0, v17" ::"f"(t5));
    t5 = a[5];
    asm volatile("vfwmacc.vf v6, %0, v17" ::"f"(t6));
    t6 = a[6];
    asm volatile("vfwmacc.vf v7, %0, v17" ::"f"(t7));
    t7 = a[7];
    asm volatile("vfwmacc.vf v8, %0, v17" ::"f"(t8));
    t8 = a[8];
    asm volatile("vfwmacc.vf v9, %0, v17" ::"f"(t9));
    t9 = a[9];
    asm volatile("vfwmacc.vf v10, %0, v17" ::"f"(t10));
    t10 = a[10];
    asm volatile("vfwmacc.vf v11, %0, v17" ::"f"(t11));
    t11 = a[11];
    asm volatile("vfwmacc.vf v12, %0, v17" ::"f"(t12));
    t12 = a[12];
    asm volatile("vfwmacc.vf v13, %0, v17" ::"f"(t13));
    t13 = a[13];
    asm volatile("vfwmacc.vf v14, %0, v17" ::"f"(t14));
    t14 = a[14];
    asm volatile("vfwmacc.vf v15, %0, v17" ::"f"(t15));
    t15 = a[15];

    a += lda;
  }

// Last iteration: final math (Ακόμα σε e16, mf2)
  asm volatile("vfwmacc.vf v0, %0, v17" ::"f"(t0));
  asm volatile("vfwmacc.vf v1, %0, v17" ::"f"(t1));
  asm volatile("vfwmacc.vf v2, %0, v17" ::"f"(t2));
  asm volatile("vfwmacc.vf v3, %0, v17" ::"f"(t3));
  asm volatile("vfwmacc.vf v4, %0, v17" ::"f"(t4));
  asm volatile("vfwmacc.vf v5, %0, v17" ::"f"(t5));
  asm volatile("vfwmacc.vf v6, %0, v17" ::"f"(t6));
  asm volatile("vfwmacc.vf v7, %0, v17" ::"f"(t7));
  asm volatile("vfwmacc.vf v8, %0, v17" ::"f"(t8));
  asm volatile("vfwmacc.vf v9, %0, v17" ::"f"(t9));
  asm volatile("vfwmacc.vf v10, %0, v17" ::"f"(t10));
  asm volatile("vfwmacc.vf v11, %0, v17" ::"f"(t11));
  asm volatile("vfwmacc.vf v12, %0, v17" ::"f"(t12));
  asm volatile("vfwmacc.vf v13, %0, v17" ::"f"(t13));
  asm volatile("vfwmacc.vf v14, %0, v17" ::"f"(t14));
  asm volatile("vfwmacc.vf v15, %0, v17" ::"f"(t15));

  // --- ΚΡΙΣΙΜΟ CONTEXT SWITCH ---
  // Γυρνάμε σε e32, m1 για να αποθηκεύσουμε τα FP32 αποτελέσματα
  asm volatile("vsetvli zero, %0, e32, m1, ta, ma" :: "r"(vl));

  asm volatile("vse32.v v0, (%0);" ::"r"(c)); c += P;
  asm volatile("vse32.v v1, (%0);" ::"r"(c)); c += P;
  asm volatile("vse32.v v2, (%0);" ::"r"(c)); c += P;
  asm volatile("vse32.v v3, (%0);" ::"r"(c)); c += P;
  asm volatile("vse32.v v4, (%0);" ::"r"(c)); c += P;
  asm volatile("vse32.v v5, (%0);" ::"r"(c)); c += P;
  asm volatile("vse32.v v6, (%0);" ::"r"(c)); c += P;
  asm volatile("vse32.v v7, (%0);" ::"r"(c)); c += P;
  asm volatile("vse32.v v8, (%0);" ::"r"(c)); c += P;
  asm volatile("vse32.v v9, (%0);" ::"r"(c)); c += P;
  asm volatile("vse32.v v10, (%0);" ::"r"(c)); c += P;
  asm volatile("vse32.v v11, (%0);" ::"r"(c)); c += P;
  asm volatile("vse32.v v12, (%0);" ::"r"(c)); c += P;
  asm volatile("vse32.v v13, (%0);" ::"r"(c)); c += P;
  asm volatile("vse32.v v14, (%0);" ::"r"(c)); c += P;
  asm volatile("vse32.v v15, (%0);" ::"r"(c));
}