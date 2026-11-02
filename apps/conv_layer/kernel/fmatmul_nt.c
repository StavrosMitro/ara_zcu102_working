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
#include "fmatmul.h"

#define MIN(a, b) ((a) < (b) ? (a) : (b))

void fmatmul_nt(float *c, const _Float16 *a, const _Float16 *b,
                const unsigned long int M, const unsigned long int N,
                const unsigned long int P) {
  if (M <= 4) {
    fmatmul_4x4_nt(c, a, b, M, N, P);
  } else if (M <= 8) {
    fmatmul_8x8_nt(c, a, b, M, N, P);
  } else if (M <= 64) {
    fmatmul_16x16_nt(c, a, b, M, N, P);
  } else if (M <= 128) {
    // With an 8x8 matmul, use LMUL=2 to increase vl.
    fmatmul_8x8_nt(c, a, b, M, N, P);
  } else {
    // With a 4x4 matmul, use LMUL=4 to increase vl.
    fmatmul_4x4_nt(c, a, b, M, N, P);
  }
}

void fmatmul_4x4_nt(float *c, const _Float16 *a, const _Float16 *b,
                    const unsigned long int M, const unsigned long int N,
                    const unsigned long int P) {
  // We work on 4 rows of the matrix at once
  const unsigned long int block_size = 4;
  unsigned long int block_size_p;

  // Set the vector configuration
  asm volatile("vsetvli %0, %1, e16, m2, ta, ma" : "=r"(block_size_p) : "r"(P));

  // Slice the matrix into a manageable number of columns p_
  for (unsigned long int p = 0; p < P; p += block_size_p) {
    // Set the vector length
    const unsigned long int p_ = MIN(P - p, block_size_p);

    // Find pointers to the submatrices
    const _Float16 *b_ = b + p * N;
    float *c_ = c + p;

    asm volatile("vsetvli zero, %0, e16, m2, ta, ma" : : "r"(p_));

    // Iterate over the rows
    for (unsigned long int m = 0; m < M; m += block_size) {
      // Find pointer to the submatrices
      const _Float16 *a_ = a + m * N;
      float *c__ = c_ + m * P;

      fmatmul_vec_4x4_slice_init(p_);
      fmatmul_vec_4x4_nt(c__, a_, b_, N, P, p_);
    }
  }
}

void fmatmul_vec_4x4_nt(float *c, const _Float16 *a, const _Float16 *b,
                        const unsigned long int N, const unsigned long int P,
                        const unsigned long int vl) {
  // Temporary variables
  _Float16 t0, t1, t2, t3;

  const _Float16 *a_ = a;
  long stride_b_bytes = (long)N * (long)sizeof(_Float16);

  asm volatile("vsetvli zero, %0, e16, m2, ta, ma" :: "r"(vl));
  // Prefetch one column of matrix B (row-major B, strided by N)
  asm volatile("vlse16.v v16, (%0), %1" : : "r"(b), "r"(stride_b_bytes));
  b += 1;

  // Prefetch one column of scalar values from A
  t0 = *a, a += N;
  t1 = *a, a += N;
  t2 = *a, a += N;
  t3 = *a;

  // Compute the multiplication
  unsigned long int n = 0;

  while (n != N) {
    // Calculate pointer to the matrix A
    a = a_ + ++n;

    asm volatile("vfwmacc.vf v0, %0, v16" ::"f"(t0));
    t0 = *a, a += N;

    // Load next column of B
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

    // Load next column of B
    asm volatile("vlse16.v v16, (%0), %1" : : "r"(b), "r"(stride_b_bytes));
    b += 1;

    asm volatile("vfwmacc.vf v4, %0, v20" ::"f"(t1));
    t1 = *a, a += N;
    asm volatile("vfwmacc.vf v8, %0, v20" ::"f"(t2));
    t2 = *a, a += N;
    asm volatile("vfwmacc.vf v12, %0, v20" ::"f"(t3));
    t3 = *a;
  }

  // Last iteration: store results
  asm volatile("vfwmacc.vf v0, %0, v20" ::"f"(t0));
  asm volatile("vse32.v v0, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfwmacc.vf v4, %0, v20" ::"f"(t1));
  asm volatile("vse32.v v4, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfwmacc.vf v8, %0, v20" ::"f"(t2));
  asm volatile("vse32.v v8, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfwmacc.vf v12, %0, v20" ::"f"(t3));
  asm volatile("vse32.v v12, (%0);" ::"r"(c));
}

void fmatmul_8x8_nt(float *c, const _Float16 *a, const _Float16 *b,
                    const unsigned long int M, const unsigned long int N,
                    const unsigned long int P) {
  // We work on 4 rows of the matrix at once
  const unsigned long int block_size = 8;
  unsigned long int block_size_p;

  // Set the vector configuration
  asm volatile("vsetvli %0, %1, e16, m1, ta, ma" : "=r"(block_size_p) : "r"(P));

  // Slice the matrix into a manageable number of columns p_
  for (unsigned long int p = 0; p < P; p += block_size_p) {
    // Set the vector length
    const unsigned long int p_ = MIN(P - p, block_size_p);

    // Find pointers to the submatrices
    const _Float16 *b_ = b + p * N;
    float *c_ = c + p;

    asm volatile("vsetvli zero, %0, e16, m1, ta, ma" : : "r"(p_));

    // Iterate over the rows
    for (unsigned long int m = 0; m < M; m += block_size) {
      // Find pointer to the submatrices
      const _Float16 *a_ = a + m * N;
      float *c__ = c_ + m * P;

      fmatmul_vec_8x8_slice_init(p_);
      fmatmul_vec_8x8_nt(c__, a_, b_, N, P, p_);
    }
  }
}

void fmatmul_vec_8x8_nt(float *c, const _Float16 *a, const _Float16 *b,
                        const unsigned long int N, const unsigned long int P,
                        const unsigned long int vl) {
  // Temporary variables
  _Float16 t0, t1, t2, t3, t4, t5, t6, t7;

  // Original pointer
  const _Float16 *a_ = a;
  long stride_b_bytes = (long)N * (long)sizeof(_Float16);

  asm volatile("vsetvli zero, %0, e16, m1, ta, ma" :: "r"(vl));
  // Prefetch one column of matrix B (row-major B, strided by N)
  asm volatile("vlse16.v v18, (%0), %1" : : "r"(b), "r"(stride_b_bytes));
  b += 1;

  // Prefetch one column of scalar values
  t0 = *a, a += N;
  t1 = *a, a += N;
  t2 = *a, a += N;
  t3 = *a, a += N;
  t4 = *a, a += N;
  t5 = *a, a += N;
  t6 = *a, a += N;
  t7 = *a;

  // Compute the multiplication
  unsigned long int n = 0;

  while (n != N) {
#ifdef VCD_DUMP
    // Start dumping VCD
    if (n == 8)
      event_trigger = +1;
    // Stop dumping VCD
    if (n == 12)
      event_trigger = -1;
#endif

    // Calculate pointer to the matrix A
    a = a_ + ++n;

    asm volatile("vfwmacc.vf v0, %0, v18" ::"f"(t0));
    t0 = *a, a += N;

    // Load next column of B
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

    // Load next column of B
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
  // Γυρνάμε τον επεξεργαστή σε 32-bit mode (m2) για να γράψει σωστά τα FP32 αποτελέσματα!
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
}

void fmatmul_16x16_nt(float *c, const _Float16 *a, const _Float16 *b,
                      unsigned long int M, unsigned long int N,
                      unsigned long int P) {
  // We work on 4 rows of the matrix at once
  const unsigned long int block_size = 16;
  unsigned long int block_size_p;

  // Set the vector configuration
  asm volatile("vsetvli %0, %1, e16, mf2, ta, ma" : "=r"(block_size_p) : "r"(P));

  // Slice the matrix into a manageable number of columns p_
  for (unsigned long int p = 0; p < P; p += block_size_p) {
    // Set the vector length
    const unsigned long int p_ = MIN(P - p, block_size_p);

    // Find pointers to the submatrices
    const _Float16 *b_ = b + p * N;
    float *c_ = c + p;

    asm volatile("vsetvli zero, %0, e16, mf2, ta, ma" : : "r"(p_));

    // Iterate over the rows
    for (unsigned long int m = 0; m < M; m += block_size) {
      // Find pointer to the submatrices
      const _Float16 *a_ = a + m * N;
      float *c__ = c_ + m * P;

      fmatmul_vec_16x16_slice_init(p_);
      fmatmul_vec_16x16_nt(c__, a_, b_, N, P, p_);
    }
  }
}

void fmatmul_vec_16x16_nt(float *c, const _Float16 *a, const _Float16 *b,
                          const unsigned long int N, const unsigned long int P,
                          const unsigned long int vl) {
  // Temporary variables
  _Float16 t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15;

  // Original pointer
  const _Float16 *a_ = a;
  long stride_b_bytes = (long)N * (long)sizeof(_Float16);

  asm volatile("vsetvli zero, %0, e16, mf2, ta, ma" :: "r"(vl));
  // Prefetch one column of scalar values
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

  // Prefetch one column of matrix B (row-major B, strided by N)
  asm volatile("vlse16.v v16, (%0), %1" : : "r"(b), "r"(stride_b_bytes));
  b += 1;

  // Compute the multiplication
  unsigned long int n = 0;

  while (n != N) {
#ifdef VCD_DUMP
    // Start dumping VCD
    if (n == 8)
      event_trigger = +1;
    // Stop dumping VCD
    if (n == 12)
      event_trigger = -1;
#endif

    // Calculate pointer to the matrix A
    a = a_ + ++n;

    asm volatile("vfwmacc.vf v0, %0, v16" ::"f"(t0));
    t0 = *a, a += N;

    // Load next column of B
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

    // Load next column of B
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

