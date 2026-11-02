// Copyright 2020 ETH Zurich and University of Bologna.
//
// SPDX-License-Identifier: Apache-2.0
//
// Author: Matheus Cavalcante / Samuel Riedel (original fmatmul_nt)
// Mixed-precision _f16out variant: Stavros Mitropoulos, NTUA
//
// fmatmul_nt_f16out — c = a * b^T with an FP16 output epilogue.
//
//   a, b   : FP16 (b accessed column-wise via strided vlse16, as in fmatmul_nt)
//   accum  : FP32 (vfwmacc.vf widening MACs)
//   c      : FP16 (_Float16*)  <-- narrowed with vfncvt.f.f.w in the store epilogue
//
// This computes the backward d_input for fc layers. d_input is FP16 because the
// layer below expects its incoming gradient (d_output) in FP16 (the FP16 gradient
// chain). d_weights/d_bias remain FP32 (fmatmul_tn is unchanged).
//
// Narrowing per block size (same VLMAX equivalence as fmatmul_nt):
//   4x4   : acc e32,m4 -> e16,m2 (v24)   [stays e16,m2, no context switch]
//   8x8   : acc e32,m2 -> e16,m1 (v16)   [stays e16,m1]
//   16x16 : acc e32,m1 -> e16,mf2 (v18)  [stays e16,mf2]

#include <stddef.h>
#include "fmatmul.h"

#define MIN(a, b) ((a) < (b) ? (a) : (b))

// ---------------------------------------------------------------------------
// 4x4
// ---------------------------------------------------------------------------
static void fmatmul_vec_4x4_nt_f16out(_Float16 *c, const _Float16 *a, const _Float16 *b,
                                      const unsigned long int N, const unsigned long int P,
                                      const unsigned long int vl) {
  _Float16 t0, t1, t2, t3;
  const _Float16 *a_ = a;
  long stride_b_bytes = (long)N * (long)sizeof(_Float16);

  asm volatile("vsetvli zero, %0, e16, m2, ta, ma" :: "r"(vl));
  asm volatile("vlse16.v v16, (%0), %1" : : "r"(b), "r"(stride_b_bytes));
  b += 1;

  t0 = *a, a += N;
  t1 = *a, a += N;
  t2 = *a, a += N;
  t3 = *a;

  unsigned long int n = 0;
  while (n != N) {
    a = a_ + ++n;

    asm volatile("vfwmacc.vf v0, %0, v16" ::"f"(t0));
    t0 = *a, a += N;
    asm volatile("vlse16.v v20, (%0), %1" : : "r"(b), "r"(stride_b_bytes));
    b += 1;
    asm volatile("vfwmacc.vf v4, %0, v16" ::"f"(t1));
    t1 = *a, a += N;
    asm volatile("vfwmacc.vf v8, %0, v16" ::"f"(t2));
    t2 = *a, a += N;
    asm volatile("vfwmacc.vf v12, %0, v16" ::"f"(t3));
    t3 = *a;

    a = a_ + ++n;
    if (n == N)
      break;

    asm volatile("vfwmacc.vf v0, %0, v20" ::"f"(t0));
    t0 = *a, a += N;
    asm volatile("vlse16.v v16, (%0), %1" : : "r"(b), "r"(stride_b_bytes));
    b += 1;
    asm volatile("vfwmacc.vf v4, %0, v20" ::"f"(t1));
    t1 = *a, a += N;
    asm volatile("vfwmacc.vf v8, %0, v20" ::"f"(t2));
    t2 = *a, a += N;
    asm volatile("vfwmacc.vf v12, %0, v20" ::"f"(t3));
    t3 = *a;
  }

  // Final MAC + narrow FP32 acc (e32,m4) -> FP16 (e16,m2, v24) + store. Stays e16,m2.
  asm volatile("vfwmacc.vf v0, %0, v20" ::"f"(t0));
  asm volatile("vfncvt.f.f.w v24, v0");
  asm volatile("vse16.v v24, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfwmacc.vf v4, %0, v20" ::"f"(t1));
  asm volatile("vfncvt.f.f.w v24, v4");
  asm volatile("vse16.v v24, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfwmacc.vf v8, %0, v20" ::"f"(t2));
  asm volatile("vfncvt.f.f.w v24, v8");
  asm volatile("vse16.v v24, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfwmacc.vf v12, %0, v20" ::"f"(t3));
  asm volatile("vfncvt.f.f.w v24, v12");
  asm volatile("vse16.v v24, (%0);" ::"r"(c));
}

void fmatmul_4x4_nt_f16out(_Float16 *c, const _Float16 *a, const _Float16 *b,
                           const unsigned long int M, const unsigned long int N,
                           const unsigned long int P) {
  const unsigned long int block_size = 4;
  unsigned long int block_size_p;

  asm volatile("vsetvli %0, %1, e16, m2, ta, ma" : "=r"(block_size_p) : "r"(P));

  for (unsigned long int p = 0; p < P; p += block_size_p) {
    const unsigned long int p_ = MIN(P - p, block_size_p);
    const _Float16 *b_ = b + p * N;
    _Float16 *c_ = c + p;

    asm volatile("vsetvli zero, %0, e16, m2, ta, ma" : : "r"(p_));

    for (unsigned long int m = 0; m < M; m += block_size) {
      const _Float16 *a_ = a + m * N;
      _Float16 *c__ = c_ + m * P;

      fmatmul_vec_4x4_slice_init(p_);
      fmatmul_vec_4x4_nt_f16out(c__, a_, b_, N, P, p_);
    }
  }
}

// ---------------------------------------------------------------------------
// 8x8
// ---------------------------------------------------------------------------
static void fmatmul_vec_8x8_nt_f16out(_Float16 *c, const _Float16 *a, const _Float16 *b,
                                      const unsigned long int N, const unsigned long int P,
                                      const unsigned long int vl) {
  _Float16 t0, t1, t2, t3, t4, t5, t6, t7;
  const _Float16 *a_ = a;
  long stride_b_bytes = (long)N * (long)sizeof(_Float16);

  asm volatile("vsetvli zero, %0, e16, m1, ta, ma" :: "r"(vl));
  asm volatile("vlse16.v v18, (%0), %1" : : "r"(b), "r"(stride_b_bytes));
  b += 1;

  t0 = *a, a += N;
  t1 = *a, a += N;
  t2 = *a, a += N;
  t3 = *a, a += N;
  t4 = *a, a += N;
  t5 = *a, a += N;
  t6 = *a, a += N;
  t7 = *a;

  unsigned long int n = 0;
  while (n != N) {
    a = a_ + ++n;

    asm volatile("vfwmacc.vf v0, %0, v18" ::"f"(t0));
    t0 = *a, a += N;
    asm volatile("vlse16.v v20, (%0), %1" : : "r"(b), "r"(stride_b_bytes));
    b += 1;
    asm volatile("vfwmacc.vf v2, %0, v18" ::"f"(t1));
    t1 = *a, a += N;
    asm volatile("vfwmacc.vf v4, %0, v18" ::"f"(t2));
    t2 = *a, a += N;
    asm volatile("vfwmacc.vf v6, %0, v18" ::"f"(t3));
    t3 = *a, a += N;
    asm volatile("vfwmacc.vf v8, %0, v18" ::"f"(t4));
    t4 = *a, a += N;
    asm volatile("vfwmacc.vf v10, %0, v18" ::"f"(t5));
    t5 = *a, a += N;
    asm volatile("vfwmacc.vf v12, %0, v18" ::"f"(t6));
    t6 = *a, a += N;
    asm volatile("vfwmacc.vf v14, %0, v18" ::"f"(t7));
    t7 = *a;

    a = a_ + ++n;
    if (n == N)
      break;

    asm volatile("vfwmacc.vf v0, %0, v20" ::"f"(t0));
    t0 = *a, a += N;
    asm volatile("vlse16.v v18, (%0), %1" : : "r"(b), "r"(stride_b_bytes));
    b += 1;
    asm volatile("vfwmacc.vf v2, %0, v20" ::"f"(t1));
    t1 = *a, a += N;
    asm volatile("vfwmacc.vf v4, %0, v20" ::"f"(t2));
    t2 = *a, a += N;
    asm volatile("vfwmacc.vf v6, %0, v20" ::"f"(t3));
    t3 = *a, a += N;
    asm volatile("vfwmacc.vf v8, %0, v20" ::"f"(t4));
    t4 = *a, a += N;
    asm volatile("vfwmacc.vf v10, %0, v20" ::"f"(t5));
    t5 = *a, a += N;
    asm volatile("vfwmacc.vf v12, %0, v20" ::"f"(t6));
    t6 = *a, a += N;
    asm volatile("vfwmacc.vf v14, %0, v20" ::"f"(t7));
    t7 = *a;
  }

  // Final MACs (still e16,m1)
  asm volatile("vfwmacc.vf v0, %0, v20" ::"f"(t0));
  asm volatile("vfwmacc.vf v2, %0, v20" ::"f"(t1));
  asm volatile("vfwmacc.vf v4, %0, v20" ::"f"(t2));
  asm volatile("vfwmacc.vf v6, %0, v20" ::"f"(t3));
  asm volatile("vfwmacc.vf v8, %0, v20" ::"f"(t4));
  asm volatile("vfwmacc.vf v10, %0, v20" ::"f"(t5));
  asm volatile("vfwmacc.vf v12, %0, v20" ::"f"(t6));
  asm volatile("vfwmacc.vf v14, %0, v20" ::"f"(t7));

  // Narrow e32,m2 -> e16,m1 (v16) and store. Stays e16,m1.
  asm volatile("vfncvt.f.f.w v16, v0");
  asm volatile("vse16.v v16, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfncvt.f.f.w v16, v2");
  asm volatile("vse16.v v16, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfncvt.f.f.w v16, v4");
  asm volatile("vse16.v v16, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfncvt.f.f.w v16, v6");
  asm volatile("vse16.v v16, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfncvt.f.f.w v16, v8");
  asm volatile("vse16.v v16, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfncvt.f.f.w v16, v10");
  asm volatile("vse16.v v16, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfncvt.f.f.w v16, v12");
  asm volatile("vse16.v v16, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfncvt.f.f.w v16, v14");
  asm volatile("vse16.v v16, (%0);" ::"r"(c));
}

void fmatmul_8x8_nt_f16out(_Float16 *c, const _Float16 *a, const _Float16 *b,
                           const unsigned long int M, const unsigned long int N,
                           const unsigned long int P) {
  const unsigned long int block_size = 8;
  unsigned long int block_size_p;

  asm volatile("vsetvli %0, %1, e16, m1, ta, ma" : "=r"(block_size_p) : "r"(P));

  for (unsigned long int p = 0; p < P; p += block_size_p) {
    const unsigned long int p_ = MIN(P - p, block_size_p);
    const _Float16 *b_ = b + p * N;
    _Float16 *c_ = c + p;

    asm volatile("vsetvli zero, %0, e16, m1, ta, ma" : : "r"(p_));

    for (unsigned long int m = 0; m < M; m += block_size) {
      const _Float16 *a_ = a + m * N;
      _Float16 *c__ = c_ + m * P;

      fmatmul_vec_8x8_slice_init(p_);
      fmatmul_vec_8x8_nt_f16out(c__, a_, b_, N, P, p_);
    }
  }
}

// ---------------------------------------------------------------------------
// 16x16
// ---------------------------------------------------------------------------
static void fmatmul_vec_16x16_nt_f16out(_Float16 *c, const _Float16 *a, const _Float16 *b,
                                        const unsigned long int N, const unsigned long int P,
                                        const unsigned long int vl) {
  _Float16 t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15;
  const _Float16 *a_ = a;
  long stride_b_bytes = (long)N * (long)sizeof(_Float16);

  asm volatile("vsetvli zero, %0, e16, mf2, ta, ma" :: "r"(vl));
  t0 = *a, a += N;
  t1 = *a, a += N;
  t2 = *a, a += N;
  t3 = *a, a += N;
  t4 = *a, a += N;
  t5 = *a, a += N;
  t6 = *a, a += N;
  t7 = *a, a += N;
  t8 = *a, a += N;
  t9 = *a, a += N;
  t10 = *a, a += N;
  t11 = *a, a += N;
  t12 = *a, a += N;
  t13 = *a, a += N;
  t14 = *a, a += N;
  t15 = *a;

  asm volatile("vlse16.v v16, (%0), %1" : : "r"(b), "r"(stride_b_bytes));
  b += 1;

  unsigned long int n = 0;
  while (n != N) {
    a = a_ + ++n;

    asm volatile("vfwmacc.vf v0, %0, v16" ::"f"(t0));
    t0 = *a, a += N;
    asm volatile("vlse16.v v17, (%0), %1" : : "r"(b), "r"(stride_b_bytes));
    b += 1;
    asm volatile("vfwmacc.vf v1, %0, v16" ::"f"(t1));
    t1 = *a, a += N;
    asm volatile("vfwmacc.vf v2, %0, v16" ::"f"(t2));
    t2 = *a, a += N;
    asm volatile("vfwmacc.vf v3, %0, v16" ::"f"(t3));
    t3 = *a, a += N;
    asm volatile("vfwmacc.vf v4, %0, v16" ::"f"(t4));
    t4 = *a, a += N;
    asm volatile("vfwmacc.vf v5, %0, v16" ::"f"(t5));
    t5 = *a, a += N;
    asm volatile("vfwmacc.vf v6, %0, v16" ::"f"(t6));
    t6 = *a, a += N;
    asm volatile("vfwmacc.vf v7, %0, v16" ::"f"(t7));
    t7 = *a, a += N;
    asm volatile("vfwmacc.vf v8, %0, v16" ::"f"(t8));
    t8 = *a, a += N;
    asm volatile("vfwmacc.vf v9, %0, v16" ::"f"(t9));
    t9 = *a, a += N;
    asm volatile("vfwmacc.vf v10, %0, v16" ::"f"(t10));
    t10 = *a, a += N;
    asm volatile("vfwmacc.vf v11, %0, v16" ::"f"(t11));
    t11 = *a, a += N;
    asm volatile("vfwmacc.vf v12, %0, v16" ::"f"(t12));
    t12 = *a, a += N;
    asm volatile("vfwmacc.vf v13, %0, v16" ::"f"(t13));
    t13 = *a, a += N;
    asm volatile("vfwmacc.vf v14, %0, v16" ::"f"(t14));
    t14 = *a, a += N;
    asm volatile("vfwmacc.vf v15, %0, v16" ::"f"(t15));
    t15 = *a;

    a = a_ + ++n;
    if (n == N)
      break;

    asm volatile("vfwmacc.vf v0, %0, v17" ::"f"(t0));
    t0 = *a, a += N;
    asm volatile("vlse16.v v16, (%0), %1" : : "r"(b), "r"(stride_b_bytes));
    b += 1;
    asm volatile("vfwmacc.vf v1, %0, v17" ::"f"(t1));
    t1 = *a, a += N;
    asm volatile("vfwmacc.vf v2, %0, v17" ::"f"(t2));
    t2 = *a, a += N;
    asm volatile("vfwmacc.vf v3, %0, v17" ::"f"(t3));
    t3 = *a, a += N;
    asm volatile("vfwmacc.vf v4, %0, v17" ::"f"(t4));
    t4 = *a, a += N;
    asm volatile("vfwmacc.vf v5, %0, v17" ::"f"(t5));
    t5 = *a, a += N;
    asm volatile("vfwmacc.vf v6, %0, v17" ::"f"(t6));
    t6 = *a, a += N;
    asm volatile("vfwmacc.vf v7, %0, v17" ::"f"(t7));
    t7 = *a, a += N;
    asm volatile("vfwmacc.vf v8, %0, v17" ::"f"(t8));
    t8 = *a, a += N;
    asm volatile("vfwmacc.vf v9, %0, v17" ::"f"(t9));
    t9 = *a, a += N;
    asm volatile("vfwmacc.vf v10, %0, v17" ::"f"(t10));
    t10 = *a, a += N;
    asm volatile("vfwmacc.vf v11, %0, v17" ::"f"(t11));
    t11 = *a, a += N;
    asm volatile("vfwmacc.vf v12, %0, v17" ::"f"(t12));
    t12 = *a, a += N;
    asm volatile("vfwmacc.vf v13, %0, v17" ::"f"(t13));
    t13 = *a, a += N;
    asm volatile("vfwmacc.vf v14, %0, v17" ::"f"(t14));
    t14 = *a, a += N;
    asm volatile("vfwmacc.vf v15, %0, v17" ::"f"(t15));
    t15 = *a;
  }

  // Final MACs (still e16,mf2)
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

  // Narrow e32,m1 -> e16,mf2 (v18) and store. Stays e16,mf2.
  asm volatile("vfncvt.f.f.w v18, v0");  asm volatile("vse16.v v18, (%0);" ::"r"(c)); c += P;
  asm volatile("vfncvt.f.f.w v18, v1");  asm volatile("vse16.v v18, (%0);" ::"r"(c)); c += P;
  asm volatile("vfncvt.f.f.w v18, v2");  asm volatile("vse16.v v18, (%0);" ::"r"(c)); c += P;
  asm volatile("vfncvt.f.f.w v18, v3");  asm volatile("vse16.v v18, (%0);" ::"r"(c)); c += P;
  asm volatile("vfncvt.f.f.w v18, v4");  asm volatile("vse16.v v18, (%0);" ::"r"(c)); c += P;
  asm volatile("vfncvt.f.f.w v18, v5");  asm volatile("vse16.v v18, (%0);" ::"r"(c)); c += P;
  asm volatile("vfncvt.f.f.w v18, v6");  asm volatile("vse16.v v18, (%0);" ::"r"(c)); c += P;
  asm volatile("vfncvt.f.f.w v18, v7");  asm volatile("vse16.v v18, (%0);" ::"r"(c)); c += P;
  asm volatile("vfncvt.f.f.w v18, v8");  asm volatile("vse16.v v18, (%0);" ::"r"(c)); c += P;
  asm volatile("vfncvt.f.f.w v18, v9");  asm volatile("vse16.v v18, (%0);" ::"r"(c)); c += P;
  asm volatile("vfncvt.f.f.w v18, v10"); asm volatile("vse16.v v18, (%0);" ::"r"(c)); c += P;
  asm volatile("vfncvt.f.f.w v18, v11"); asm volatile("vse16.v v18, (%0);" ::"r"(c)); c += P;
  asm volatile("vfncvt.f.f.w v18, v12"); asm volatile("vse16.v v18, (%0);" ::"r"(c)); c += P;
  asm volatile("vfncvt.f.f.w v18, v13"); asm volatile("vse16.v v18, (%0);" ::"r"(c)); c += P;
  asm volatile("vfncvt.f.f.w v18, v14"); asm volatile("vse16.v v18, (%0);" ::"r"(c)); c += P;
  asm volatile("vfncvt.f.f.w v18, v15"); asm volatile("vse16.v v18, (%0);" ::"r"(c));
}

void fmatmul_16x16_nt_f16out(_Float16 *c, const _Float16 *a, const _Float16 *b,
                             unsigned long int M, unsigned long int N,
                             unsigned long int P) {
  const unsigned long int block_size = 16;
  unsigned long int block_size_p;

  asm volatile("vsetvli %0, %1, e16, mf2, ta, ma" : "=r"(block_size_p) : "r"(P));

  for (unsigned long int p = 0; p < P; p += block_size_p) {
    const unsigned long int p_ = MIN(P - p, block_size_p);
    const _Float16 *b_ = b + p * N;
    _Float16 *c_ = c + p;

    asm volatile("vsetvli zero, %0, e16, mf2, ta, ma" : : "r"(p_));

    for (unsigned long int m = 0; m < M; m += block_size) {
      const _Float16 *a_ = a + m * N;
      _Float16 *c__ = c_ + m * P;

      fmatmul_vec_16x16_slice_init(p_);
      fmatmul_vec_16x16_nt_f16out(c__, a_, b_, N, P, p_);
    }
  }
}

// Dispatcher — same M-based block selection as fmatmul_nt.
void fmatmul_nt_f16out(_Float16 *c, const _Float16 *a, const _Float16 *b,
                       const unsigned long int M, const unsigned long int N,
                       const unsigned long int P) {
  if (M <= 4) {
    fmatmul_4x4_nt_f16out(c, a, b, M, N, P);
  } else if (M <= 8) {
    fmatmul_8x8_nt_f16out(c, a, b, M, N, P);
  } else if (M <= 64) {
    fmatmul_16x16_nt_f16out(c, a, b, M, N, P);
  } else if (M <= 128) {
    fmatmul_8x8_nt_f16out(c, a, b, M, N, P);
  } else {
    fmatmul_4x4_nt_f16out(c, a, b, M, N, P);
  }
}
