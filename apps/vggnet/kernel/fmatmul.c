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
// modified versions of fmatmul_32 Author: Stavros Mitropoulos, NTUA

#include <stddef.h>
#include "fmatmul.h"

#define MIN(a, b) ((a) < (b) ? (a) : (b))

void fmatmul_32(float *c, const float *a, const float *b,
             const unsigned long int M, const unsigned long int N,
             const unsigned long int P) {
  if (M <= 4) {
    fmatmul_4x4_32(c, a, b, M, N, P);
  } else if (M <= 8) {
    fmatmul_8x8_32(c, a, b, M, N, P);
  } else if (M <= 64) {
    fmatmul_16x16_32(c, a, b, M, N, P);
  } else if (M <= 128) {
    // With an 8x8 matmul, use LMUL=2 to increase vl.
    fmatmul_8x8_32(c, a, b, M, N, P);
  } else {
    // With a 4x4 matmul, use LMUL=4 to increase vl.
    fmatmul_4x4_32(c, a, b, M, N, P);
  }
}

void fmatmul_fused_32(float *c, const float *a, const float *b,
                   const float *bias, const unsigned long int M,
                   const unsigned long int N, const unsigned long int P) {
  if (M <= 4) {
    fmatmul_4x4_fused_32(c, a, b, bias, M, N, P);
  } else if (M <= 8) {
    fmatmul_8x8_fused_32(c, a, b, bias, M, N, P);
  } else if (M <= 64) {
    fmatmul_16x16_fused_32(c, a, b, bias, M, N, P);
  } else if (M <= 128) {
    // With an 8x8 matmul, use LMUL=2 to increase vl.
    fmatmul_8x8_fused_32(c, a, b, bias, M, N, P);
  } else {
    // With a 4x4 matmul, use LMUL=4 to increase vl.
    fmatmul_4x4_fused_32(c, a, b, bias, M, N, P);
  }
}

void fmatmul_nt_32(float *c, const float *a, const float *b,
                const unsigned long int M, const unsigned long int N,
                const unsigned long int P) {
  if (M <= 4) {
    fmatmul_4x4_nt_32(c, a, b, M, N, P);
  } else if (M <= 8) {
    fmatmul_8x8_nt_32(c, a, b, M, N, P);
  } else if (M <= 64) {
    fmatmul_16x16_nt_32(c, a, b, M, N, P);
  } else if (M <= 128) {
    // With an 8x8 matmul, use LMUL=2 to increase vl.
    fmatmul_8x8_nt_32(c, a, b, M, N, P);
  } else {
    // With a 4x4 matmul, use LMUL=4 to increase vl.
    fmatmul_4x4_nt_32(c, a, b, M, N, P);
  }
}

void fmatmul_tn_32(float *c, const float *a, const float *b,
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

  // NOTE: this is a pure c = a^T * b. It used to divide the result by M, which
  // made a general-purpose matmul silently behave as a batch-mean reducer --
  // easy to miss at the call site. The 1/batchsize now lives once in
  // cross_entropy_loss(). Other apps (fc_layer32, fc_layer16only, fmatmul_cmp)
  // carry their own copies of this file and are deliberately left as they were.
}

// ---------------
// 4x4
// ---------------

void fmatmul_4x4_32(float *c, const float *a, const float *b,
                 const unsigned long int M, const unsigned long int N,
                 const unsigned long int P) {
  // We work on 4 rows of the matrix at once
  const unsigned long int block_size = 4;
  unsigned long int block_size_p;

  // Set the vector configuration
  asm volatile("vsetvli %0, %1, e32, m4, ta, ma" : "=r"(block_size_p) : "r"(P)); 
  /*
  block_size_p --> actual vector length
  vsetvli returns min(input, VLMAX)
  */

  // Slice the matrix into a manageable number of columns p_
  for (unsigned long int p = 0; p < P; p += block_size_p) {
    // Set the vector length
    const unsigned long int p_ = MIN(P - p, block_size_p);

    // Find pointers to the submatrices
    const float *b_ = b + p;
    float *c_ = c + p;

    asm volatile("vsetvli zero, %0, e32, m4, ta, ma" ::"r"(p_));

    // Iterate over the rows
    for (unsigned long int m = 0; m < M; m += block_size) {
      // Find pointer to the submatrices
      const float *a_ = a + m * N;
      float *c__ = c_ + m * P;

      fmatmul_vec_4x4_slice_init_32();
      fmatmul_vec_4x4_32(c__, a_, b_, N, P);
    }
  }
}

void fmatmul_4x4_fused_32(float *c, const float *a, const float *b,
                       const float *bias, const unsigned long int M,
                       const unsigned long int N, const unsigned long int P) {
  // We work on 4 rows of the matrix at once
  const unsigned long int block_size = 4;
  unsigned long int block_size_p;

  // Set the vector configuration
  asm volatile("vsetvli %0, %1, e32, m4, ta, ma" : "=r"(block_size_p) : "r"(P));

  // Slice the matrix into a manageable number of columns p_
  for (unsigned long int p = 0; p < P; p += block_size_p) {
    // Set the vector length
    const unsigned long int p_ = MIN(P - p, block_size_p);

    // Find pointers to the submatrices
    const float *b_ = b + p;
    float *c_ = c + p;
    const float *bias_slice = bias + p;

    asm volatile("vsetvli zero, %0, e32, m4, ta, ma" : : "r"(p_));

    // Iterate over the rows
    for (unsigned long int m = 0; m < M; m += block_size) {
      // Find pointer to the submatrices
      const float *a_ = a + m * N;
      float *c__ = c_ + m * P;

      fmatmul_vec_4x4_slice_init_fused_32(bias_slice);
      fmatmul_vec_4x4_32(c__, a_, b_, N, P);
    }
  }
}

void fmatmul_4x4_nt_32(float *c, const float *a, const float *b,
                    const unsigned long int M, const unsigned long int N,
                    const unsigned long int P) {
  // We work on 4 rows of the matrix at once
  const unsigned long int block_size = 4;
  unsigned long int block_size_p;

  // Set the vector configuration
  asm volatile("vsetvli %0, %1, e32, m4, ta, ma" : "=r"(block_size_p) : "r"(P));

  // Slice the matrix into a manageable number of columns p_
  for (unsigned long int p = 0; p < P; p += block_size_p) {
    // Set the vector length
    const unsigned long int p_ = MIN(P - p, block_size_p);

    // Find pointers to the submatrices
    const float *b_ = b + p * N;
    float *c_ = c + p;

    asm volatile("vsetvli zero, %0, e32, m4, ta, ma" : : "r"(p_));

    // Iterate over the rows
    for (unsigned long int m = 0; m < M; m += block_size) {
      // Find pointer to the submatrices
      const float *a_ = a + m * N;
      float *c__ = c_ + m * P;

      fmatmul_vec_4x4_slice_init_32();
      fmatmul_vec_4x4_nt_32(c__, a_, b_, N, P);
    }
  }
}

void fmatmul_4x4_tn_32(float *c, const float *a, const float *b,
                    const unsigned long int M, const unsigned long int N,
                    const unsigned long int P) {
  const unsigned long int block_size = 4;
  unsigned long int block_size_p;

  asm volatile("vsetvli %0, %1, e32, m4, ta, ma" : "=r"(block_size_p) : "r"(P));

  for (unsigned long int p = 0; p < P; p += block_size_p) {
    const unsigned long int p_ = MIN(P - p, block_size_p);

    const float *b_ = b + p;
    float *c_ = c + p;

    asm volatile("vsetvli zero, %0, e32, m4, ta, ma" : : "r"(p_));

    for (unsigned long int n = 0; n < N; n += block_size) {
      const float *a_ = a + n;
      float *c__ = c_ + n * P;

      fmatmul_vec_4x4_slice_init_32();
      fmatmul_vec_4x4_tn_32(c__, a_, b_, M, P, N);
    }
  }
}

// ---------------
// 4x4 deferred (A * B^T)
// ---------------

static inline void fmatmul_vec_4x4_deferred(float *c, const float *a,
                                            const float *b,
                                            const unsigned long int N,
                                            const unsigned long int ldc) {
  const float *a0 = a;
  const float *a1 = a + N;
  const float *a2 = a + 2 * N;
  const float *a3 = a + 3 * N;

  const float *b0 = b;
  const float *b1 = b + N;
  const float *b2 = b + 2 * N;
  const float *b3 = b + 3 * N;

  asm volatile("vsetvli zero, %0, e32, m1, ta, ma" : : "r"(N));
  asm volatile("vmv.v.i v16, 0");
  asm volatile("vmv.v.i v17, 0");
  asm volatile("vmv.v.i v18, 0");
  asm volatile("vmv.v.i v19, 0");
  asm volatile("vmv.v.i v20, 0");
  asm volatile("vmv.v.i v21, 0");
  asm volatile("vmv.v.i v22, 0");
  asm volatile("vmv.v.i v23, 0");
  asm volatile("vmv.v.i v24, 0");
  asm volatile("vmv.v.i v25, 0");
  asm volatile("vmv.v.i v26, 0");
  asm volatile("vmv.v.i v27, 0");
  asm volatile("vmv.v.i v28, 0");
  asm volatile("vmv.v.i v29, 0");
  asm volatile("vmv.v.i v30, 0");
  asm volatile("vmv.v.i v31, 0");

  unsigned long int k = N;
  while (k > 0) {
    unsigned long int vl;
    asm volatile("vsetvli %0, %1, e32, m1, ta, ma" : "=r"(vl) : "r"(k));

    asm volatile("vle32.v v0, (%0)" : : "r"(a0));
    a0 += vl;
    asm volatile("vle32.v v1, (%0)" : : "r"(a1));
    a1 += vl;
    asm volatile("vle32.v v2, (%0)" : : "r"(a2));
    a2 += vl;
    asm volatile("vle32.v v3, (%0)" : : "r"(a3));
    a3 += vl;

    asm volatile("vle32.v v4, (%0)" : : "r"(b0));
    b0 += vl;
    asm volatile("vle32.v v5, (%0)" : : "r"(b1));
    b1 += vl;
    asm volatile("vle32.v v6, (%0)" : : "r"(b2));
    b2 += vl;
    asm volatile("vle32.v v7, (%0)" : : "r"(b3));
    b3 += vl;

    asm volatile("vfmacc.vv v16, v0, v4");
    asm volatile("vfmacc.vv v17, v0, v5");
    asm volatile("vfmacc.vv v18, v0, v6");
    asm volatile("vfmacc.vv v19, v0, v7");

    asm volatile("vfmacc.vv v20, v1, v4");
    asm volatile("vfmacc.vv v21, v1, v5");
    asm volatile("vfmacc.vv v22, v1, v6");
    asm volatile("vfmacc.vv v23, v1, v7");

    asm volatile("vfmacc.vv v24, v2, v4");
    asm volatile("vfmacc.vv v25, v2, v5");
    asm volatile("vfmacc.vv v26, v2, v6");
    asm volatile("vfmacc.vv v27, v2, v7");

    asm volatile("vfmacc.vv v28, v3, v4");
    asm volatile("vfmacc.vv v29, v3, v5");
    asm volatile("vfmacc.vv v30, v3, v6");
    asm volatile("vfmacc.vv v31, v3, v7");

    k -= vl;
  }

  asm volatile("vsetvli zero, %0, e32, m1, ta, ma" : : "r"(N));
  asm volatile("vmv.v.i v8, 0");

  asm volatile("vredsum.vs v16, v16, v8");
  asm volatile("vredsum.vs v17, v17, v8");
  asm volatile("vredsum.vs v18, v18, v8");
  asm volatile("vredsum.vs v19, v19, v8");
  asm volatile("vredsum.vs v20, v20, v8");
  asm volatile("vredsum.vs v21, v21, v8");
  asm volatile("vredsum.vs v22, v22, v8");
  asm volatile("vredsum.vs v23, v23, v8");
  asm volatile("vredsum.vs v24, v24, v8");
  asm volatile("vredsum.vs v25, v25, v8");
  asm volatile("vredsum.vs v26, v26, v8");
  asm volatile("vredsum.vs v27, v27, v8");
  asm volatile("vredsum.vs v28, v28, v8");
  asm volatile("vredsum.vs v29, v29, v8");
  asm volatile("vredsum.vs v30, v30, v8");
  asm volatile("vredsum.vs v31, v31, v8");

  float *c_ptr = c;
  float res;

  asm volatile("vfmv.f.s %0, v16" : "=f"(res));
  c_ptr[0] += res;
  asm volatile("vfmv.f.s %0, v17" : "=f"(res));
  c_ptr[1] += res;
  asm volatile("vfmv.f.s %0, v18" : "=f"(res));
  c_ptr[2] += res;
  asm volatile("vfmv.f.s %0, v19" : "=f"(res));
  c_ptr[3] += res;

  c_ptr += ldc;
  asm volatile("vfmv.f.s %0, v20" : "=f"(res));
  c_ptr[0] += res;
  asm volatile("vfmv.f.s %0, v21" : "=f"(res));
  c_ptr[1] += res;
  asm volatile("vfmv.f.s %0, v22" : "=f"(res));
  c_ptr[2] += res;
  asm volatile("vfmv.f.s %0, v23" : "=f"(res));
  c_ptr[3] += res;

  c_ptr += ldc;
  asm volatile("vfmv.f.s %0, v24" : "=f"(res));
  c_ptr[0] += res;
  asm volatile("vfmv.f.s %0, v25" : "=f"(res));
  c_ptr[1] += res;
  asm volatile("vfmv.f.s %0, v26" : "=f"(res));
  c_ptr[2] += res;
  asm volatile("vfmv.f.s %0, v27" : "=f"(res));
  c_ptr[3] += res;

  c_ptr += ldc;
  asm volatile("vfmv.f.s %0, v28" : "=f"(res));
  c_ptr[0] += res;
  asm volatile("vfmv.f.s %0, v29" : "=f"(res));
  c_ptr[1] += res;
  asm volatile("vfmv.f.s %0, v30" : "=f"(res));
  c_ptr[2] += res;
  asm volatile("vfmv.f.s %0, v31" : "=f"(res));
  c_ptr[3] += res;
}

void fmatmul_4x4_deferred_32(float *c, const float *a, const float *b,
                          const unsigned long int M, const unsigned long int N,
                          const unsigned long int P) {
  // C = A * B^T
  // A: [M x N], B: [P x N], C: [M x P]
  for (unsigned long int m = 0; m < M; m += 4) {
    for (unsigned long int p = 0; p < P; p += 4) {
      const float *a_block = a + m * N;
      const float *b_block = b + p * N;
      float *c_block = c + m * P + p;

      fmatmul_vec_4x4_deferred(c_block, a_block, b_block, N, P);
    }
  }
}

void fmatmul_vec_4x4_slice_init_32() {
  asm volatile("vmv.v.i v0,  0");
  asm volatile("vmv.v.i v4,  0");
  asm volatile("vmv.v.i v8,  0");
  asm volatile("vmv.v.i v12, 0");
}

void fmatmul_vec_4x4_slice_load_bias_32(const float *bias_slice) {
  asm volatile("vle32.v v0, (%0);" ::"r"(bias_slice));
  asm volatile("vmv.v.v v4,  v0");
  asm volatile("vmv.v.v v8,  v0");
  asm volatile("vmv.v.v v12, v0");
}

void fmatmul_vec_4x4_slice_init_fused_32(const float *bias_slice) {
  asm volatile("vle32.v v0, (%0);" ::"r"(bias_slice));
  asm volatile("vmv.v.v v4,  v0");
  asm volatile("vmv.v.v v8,  v0");
  asm volatile("vmv.v.v v12, v0");
}

void fmatmul_vec_4x4_32(float *c, const float *a, const float *b,
                     const unsigned long int N, const unsigned long int P) {
  // Temporary variables
  float t0, t1, t2, t3;

  // Original pointer
  const float *a_ = a;

  // Prefetch one row of matrix B
  asm volatile("vle32.v v16, (%0);" ::"r"(b));
  b += P;

  // Prefetch one row of scalar values
  t0 = *a, a += N;
  t1 = *a, a += N;
  t2 = *a, a += N;
  t3 = *a;

  // Compute the multiplication
  unsigned long int n = 0;

  while (n != N) {


    // Calculate pointer to the matrix A
    a = a_ + ++n;

    asm volatile("vfmacc.vf v0, %0, v16" ::"f"(t0));
    t0 = *a, a += N;

    // Load one row of B
    asm volatile("vle32.v v20, (%0);" ::"r"(b));
    b += P;

    asm volatile("vfmacc.vf v4, %0, v16" ::"f"(t1));
    t1 = *a, a += N;
    asm volatile("vfmacc.vf v8, %0, v16" ::"f"(t2));
    t2 = *a, a += N;
    asm volatile("vfmacc.vf v12, %0, v16" ::"f"(t3));
    t3 = *a;

    a = a_ + ++n;

    if (n == N)
      break;

    asm volatile("vfmacc.vf v0, %0, v20" ::"f"(t0));
    t0 = *a, a += N;

    // Load one row of B
    asm volatile("vle32.v v16, (%0);" ::"r"(b));
    b += P;

    asm volatile("vfmacc.vf v4, %0, v20" ::"f"(t1));
    t1 = *a, a += N;
    asm volatile("vfmacc.vf v8, %0, v20" ::"f"(t2));
    t2 = *a, a += N;
    asm volatile("vfmacc.vf v12, %0, v20" ::"f"(t3));
    t3 = *a;
  }

  // Last iteration: store results
  asm volatile("vfmacc.vf v0, %0, v20" ::"f"(t0));
  asm volatile("vse32.v v0, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfmacc.vf v4, %0, v20" ::"f"(t1));
  asm volatile("vse32.v v4, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfmacc.vf v8, %0, v20" ::"f"(t2));
  asm volatile("vse32.v v8, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfmacc.vf v12, %0, v20" ::"f"(t3));
  asm volatile("vse32.v v12, (%0);" ::"r"(c));
}

void fmatmul_vec_4x4_nt_32(float *c, const float *a, const float *b,
                        const unsigned long int N, const unsigned long int P) {
  // Temporary variables
  float t0, t1, t2, t3;

  const float *a_ = a;
  long stride_b_bytes = (long)N * (long)sizeof(float);

  // Prefetch one column of matrix B (row-major B, strided by N)
  asm volatile("vlse32.v v16, (%0), %1" : : "r"(b), "r"(stride_b_bytes));
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

    asm volatile("vfmacc.vf v0, %0, v16" ::"f"(t0));
    t0 = *a, a += N;

    // Load next column of B
    asm volatile("vlse32.v v20, (%0), %1" : : "r"(b), "r"(stride_b_bytes));
    b += 1;

    asm volatile("vfmacc.vf v4, %0, v16" ::"f"(t1));
    t1 = *a, a += N;
    asm volatile("vfmacc.vf v8, %0, v16" ::"f"(t2));
    t2 = *a, a += N;
    asm volatile("vfmacc.vf v12, %0, v16" ::"f"(t3));
    t3 = *a;

    a = a_ + ++n;

    if (n == N)
      break;

    asm volatile("vfmacc.vf v0, %0, v20" ::"f"(t0));
    t0 = *a, a += N;

    // Load next column of B
    asm volatile("vlse32.v v16, (%0), %1" : : "r"(b), "r"(stride_b_bytes));
    b += 1;

    asm volatile("vfmacc.vf v4, %0, v20" ::"f"(t1));
    t1 = *a, a += N;
    asm volatile("vfmacc.vf v8, %0, v20" ::"f"(t2));
    t2 = *a, a += N;
    asm volatile("vfmacc.vf v12, %0, v20" ::"f"(t3));
    t3 = *a;
  }

  // Last iteration: store results
  asm volatile("vfmacc.vf v0, %0, v20" ::"f"(t0));
  asm volatile("vse32.v v0, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfmacc.vf v4, %0, v20" ::"f"(t1));
  asm volatile("vse32.v v4, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfmacc.vf v8, %0, v20" ::"f"(t2));
  asm volatile("vse32.v v8, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfmacc.vf v12, %0, v20" ::"f"(t3));
  asm volatile("vse32.v v12, (%0);" ::"r"(c));
}

void fmatmul_vec_4x4_tn_32(float *c, const float *a, const float *b,
                        const unsigned long int N, const unsigned long int P,
                        const unsigned long int lda) { //could use smaller LMUL=1/2
  float t0, t1, t2, t3;

  if (N <= 2) { //batchsize=2
    const float *a_row = a;
    for (unsigned long int m = 0; m < N; m++) {
      asm volatile("vle32.v v16, (%0);" ::"r"(b));
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
  asm volatile("vle32.v v16, (%0);" ::"r"(b));
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

    asm volatile("vle32.v v20, (%0);" ::"r"(b));
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

    asm volatile("vle32.v v16, (%0);" ::"r"(b));
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
  asm volatile("vse32.v v0, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfmacc.vf v4, %0, v20" ::"f"(t1));
  asm volatile("vse32.v v4, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfmacc.vf v8, %0, v20" ::"f"(t2));
  asm volatile("vse32.v v8, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfmacc.vf v12, %0, v20" ::"f"(t3));
  asm volatile("vse32.v v12, (%0);" ::"r"(c));
}

// ---------------
// 8x8
// ---------------

void fmatmul_8x8_32(float *c, const float *a, const float *b,
                 const unsigned long int M, const unsigned long int N,
                 const unsigned long int P) {
  // We work on 4 rows of the matrix at once
  const unsigned long int block_size = 8;
  unsigned long int block_size_p;

  // Set the vector configuration
  asm volatile("vsetvli %0, %1, e32, m2, ta, ma" : "=r"(block_size_p) : "r"(P));

  // Slice the matrix into a manageable number of columns p_
  for (unsigned long int p = 0; p < P; p += block_size_p) {
    // Set the vector length
    const unsigned long int p_ = MIN(P - p, block_size_p);

    // Find pointers to the submatrices
    const float *b_ = b + p;
    float *c_ = c + p;

    asm volatile("vsetvli zero, %0, e32, m2, ta, ma" ::"r"(p_));

    // Iterate over the rows
    for (unsigned long int m = 0; m < M; m += block_size) {
      // Find pointer to the submatrices
      const float *a_ = a + m * N;
      float *c__ = c_ + m * P;

      fmatmul_vec_8x8_slice_init_32();
      fmatmul_vec_8x8_32(c__, a_, b_, N, P);
    }
  }
}

void fmatmul_8x8_fused_32(float *c, const float *a, const float *b,
                       const float *bias, const unsigned long int M,
                       const unsigned long int N, const unsigned long int P) {
  // We work on 4 rows of the matrix at once
  const unsigned long int block_size = 8;
  unsigned long int block_size_p;

  // Set the vector configuration
  asm volatile("vsetvli %0, %1, e32, m2, ta, ma" : "=r"(block_size_p) : "r"(P));

  // Slice the matrix into a manageable number of columns p_
  for (unsigned long int p = 0; p < P; p += block_size_p) {
    // Set the vector length
    const unsigned long int p_ = MIN(P - p, block_size_p);

    // Find pointers to the submatrices
    const float *b_ = b + p;
    float *c_ = c + p;
    const float *bias_slice = bias + p;

    asm volatile("vsetvli zero, %0, e32, m2, ta, ma" : : "r"(p_));

    // Iterate over the rows
    for (unsigned long int m = 0; m < M; m += block_size) {
      // Find pointer to the submatrices
      const float *a_ = a + m * N;
      float *c__ = c_ + m * P;

      fmatmul_vec_8x8_slice_init_fused_32(bias_slice);
      fmatmul_vec_8x8_32(c__, a_, b_, N, P);
    }
  }
}

void fmatmul_8x8_nt_32(float *c, const float *a, const float *b,
                    const unsigned long int M, const unsigned long int N,
                    const unsigned long int P) {
  // We work on 4 rows of the matrix at once
  const unsigned long int block_size = 8;
  unsigned long int block_size_p;

  // Set the vector configuration
  asm volatile("vsetvli %0, %1, e32, m2, ta, ma" : "=r"(block_size_p) : "r"(P));

  // Slice the matrix into a manageable number of columns p_
  for (unsigned long int p = 0; p < P; p += block_size_p) {
    // Set the vector length
    const unsigned long int p_ = MIN(P - p, block_size_p);

    // Find pointers to the submatrices
    const float *b_ = b + p * N;
    float *c_ = c + p;

    asm volatile("vsetvli zero, %0, e32, m2, ta, ma" : : "r"(p_));

    // Iterate over the rows
    for (unsigned long int m = 0; m < M; m += block_size) {
      // Find pointer to the submatrices
      const float *a_ = a + m * N;
      float *c__ = c_ + m * P;

      fmatmul_vec_8x8_slice_init_32();
      fmatmul_vec_8x8_nt_32(c__, a_, b_, N, P);
    }
  }
}

void fmatmul_8x8_tn_32(float *c, const float *a, const float *b,
                    const unsigned long int M, const unsigned long int N,
                    const unsigned long int P) {
  const unsigned long int block_size = 8;
  unsigned long int block_size_p;

  asm volatile("vsetvli %0, %1, e32, m2, ta, ma" : "=r"(block_size_p) : "r"(P));

  for (unsigned long int p = 0; p < P; p += block_size_p) {
    const unsigned long int p_ = MIN(P - p, block_size_p);

    const float *b_ = b + p;
    float *c_ = c + p;

    asm volatile("vsetvli zero, %0, e32, m2, ta, ma" : : "r"(p_));

    for (unsigned long int n = 0; n < N; n += block_size) {
      const float *a_ = a + n;
      float *c__ = c_ + n * P;

      fmatmul_vec_8x8_slice_init_32();
      fmatmul_vec_8x8_tn_32(c__, a_, b_, M, P, N);
    }
  }
}

void fmatmul_vec_8x8_slice_init_32() {
  asm volatile("vmv.v.i v0,  0");
  asm volatile("vmv.v.i v2,  0");
  asm volatile("vmv.v.i v4,  0");
  asm volatile("vmv.v.i v6,  0");
  asm volatile("vmv.v.i v8,  0");
  asm volatile("vmv.v.i v10, 0");
  asm volatile("vmv.v.i v12, 0");
  asm volatile("vmv.v.i v14, 0");
}

void fmatmul_vec_8x8_slice_load_bias_32(const float *bias_slice) {
  asm volatile("vle32.v v0, (%0);" ::"r"(bias_slice));
  asm volatile("vmv.v.v v2,  v0");
  asm volatile("vmv.v.v v4,  v0");
  asm volatile("vmv.v.v v6,  v0");
  asm volatile("vmv.v.v v8,  v0");
  asm volatile("vmv.v.v v10, v0");
  asm volatile("vmv.v.v v12, v0");
  asm volatile("vmv.v.v v14, v0");
}

void fmatmul_vec_8x8_slice_init_fused_32(const float *bias_slice) {
  asm volatile("vle32.v v0, (%0);" ::"r"(bias_slice));
  asm volatile("vmv.v.v v2,  v0");
  asm volatile("vmv.v.v v4,  v0");
  asm volatile("vmv.v.v v6,  v0");
  asm volatile("vmv.v.v v8,  v0");
  asm volatile("vmv.v.v v10, v0");
  asm volatile("vmv.v.v v12, v0");
  asm volatile("vmv.v.v v14, v0");
}

void fmatmul_vec_8x8_32(float *c, const float *a, const float *b,
                     const unsigned long int N, const unsigned long int P) {
  // Temporary variables
  float t0, t1, t2, t3, t4, t5, t6, t7;

  // Original pointer
  const float *a_ = a;

  // Prefetch one row of matrix B
  asm volatile("vle32.v v18, (%0);" ::"r"(b));
  b += P;

  // Prefetch one row of scalar values
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

    asm volatile("vfmacc.vf v0, %0, v18" ::"f"(t0));
    t0 = *a, a += N;

    // Load one row of B
    asm volatile("vle32.v v20, (%0);" ::"r"(b));
    b += P;

    asm volatile("vfmacc.vf v2, %0, v18" ::"f"(t1));
    t1 = *a, a += N;
    asm volatile("vfmacc.vf v4, %0, v18" ::"f"(t2));
    t2 = *a, a += N;
    asm volatile("vfmacc.vf v6, %0, v18" ::"f"(t3));
    t3 = *a, a += N;
    asm volatile("vfmacc.vf v8, %0, v18" ::"f"(t4));
    t4 = *a, a += N;
    asm volatile("vfmacc.vf v10, %0, v18" ::"f"(t5));
    t5 = *a, a += N;
    asm volatile("vfmacc.vf v12, %0, v18" ::"f"(t6));
    t6 = *a, a += N;
    asm volatile("vfmacc.vf v14, %0, v18" ::"f"(t7));
    t7 = *a;

    a = a_ + ++n;

    if (n == N)
      break;

    asm volatile("vfmacc.vf v0, %0, v20" ::"f"(t0));
    t0 = *a, a += N;

    // Load one row of B
    asm volatile("vle32.v v18, (%0);" ::"r"(b));
    b += P;

    asm volatile("vfmacc.vf v2, %0, v20" ::"f"(t1));
    t1 = *a, a += N;
    asm volatile("vfmacc.vf v4, %0, v20" ::"f"(t2));
    t2 = *a, a += N;
    asm volatile("vfmacc.vf v6, %0, v20" ::"f"(t3));
    t3 = *a, a += N;
    asm volatile("vfmacc.vf v8, %0, v20" ::"f"(t4));
    t4 = *a, a += N;
    asm volatile("vfmacc.vf v10, %0, v20" ::"f"(t5));
    t5 = *a, a += N;
    asm volatile("vfmacc.vf v12, %0, v20" ::"f"(t6));
    t6 = *a, a += N;
    asm volatile("vfmacc.vf v14, %0, v20" ::"f"(t7));
    t7 = *a;
  }

  // Last iteration: store results
  asm volatile("vfmacc.vf v0, %0, v20" ::"f"(t0));
  asm volatile("vse32.v v0, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfmacc.vf v2, %0, v20" ::"f"(t1));
  asm volatile("vse32.v v2, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfmacc.vf v4, %0, v20" ::"f"(t2));
  asm volatile("vse32.v v4, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfmacc.vf v6, %0, v20" ::"f"(t3));
  asm volatile("vse32.v v6, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfmacc.vf v8, %0, v20" ::"f"(t4));
  asm volatile("vse32.v v8, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfmacc.vf v10, %0, v20" ::"f"(t5));
  asm volatile("vse32.v v10, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfmacc.vf v12, %0, v20" ::"f"(t6));
  asm volatile("vse32.v v12, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfmacc.vf v14, %0, v20" ::"f"(t7));
  asm volatile("vse32.v v14, (%0);" ::"r"(c));
}

void fmatmul_vec_8x8_nt_32(float *c, const float *a, const float *b,
                        const unsigned long int N, const unsigned long int P) {
  // Temporary variables
  float t0, t1, t2, t3, t4, t5, t6, t7;

  // Original pointer
  const float *a_ = a;
  long stride_b_bytes = (long)N * (long)sizeof(float);

  // Prefetch one column of matrix B (row-major B, strided by N)
  asm volatile("vlse32.v v18, (%0), %1" : : "r"(b), "r"(stride_b_bytes));
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

    asm volatile("vfmacc.vf v0, %0, v18" ::"f"(t0));
    t0 = *a, a += N;

    // Load next column of B
    asm volatile("vlse32.v v20, (%0), %1" : : "r"(b), "r"(stride_b_bytes));
    b += 1;

    asm volatile("vfmacc.vf v2, %0, v18" ::"f"(t1));
    t1 = *a, a += N;
    asm volatile("vfmacc.vf v4, %0, v18" ::"f"(t2));
    t2 = *a, a += N;
    asm volatile("vfmacc.vf v6, %0, v18" ::"f"(t3));
    t3 = *a, a += N;
    asm volatile("vfmacc.vf v8, %0, v18" ::"f"(t4));
    t4 = *a, a += N;
    asm volatile("vfmacc.vf v10, %0, v18" ::"f"(t5));
    t5 = *a, a += N;
    asm volatile("vfmacc.vf v12, %0, v18" ::"f"(t6));
    t6 = *a, a += N;
    asm volatile("vfmacc.vf v14, %0, v18" ::"f"(t7));
    t7 = *a;

    a = a_ + ++n;

    if (n == N)
      break;

    asm volatile("vfmacc.vf v0, %0, v20" ::"f"(t0));
    t0 = *a, a += N;

    // Load next column of B
    asm volatile("vlse32.v v18, (%0), %1" : : "r"(b), "r"(stride_b_bytes));
    b += 1;

    asm volatile("vfmacc.vf v2, %0, v20" ::"f"(t1));
    t1 = *a, a += N;
    asm volatile("vfmacc.vf v4, %0, v20" ::"f"(t2));
    t2 = *a, a += N;
    asm volatile("vfmacc.vf v6, %0, v20" ::"f"(t3));
    t3 = *a, a += N;
    asm volatile("vfmacc.vf v8, %0, v20" ::"f"(t4));
    t4 = *a, a += N;
    asm volatile("vfmacc.vf v10, %0, v20" ::"f"(t5));
    t5 = *a, a += N;
    asm volatile("vfmacc.vf v12, %0, v20" ::"f"(t6));
    t6 = *a, a += N;
    asm volatile("vfmacc.vf v14, %0, v20" ::"f"(t7));
    t7 = *a;
  }

  // Last iteration: store results
  asm volatile("vfmacc.vf v0, %0, v20" ::"f"(t0));
  asm volatile("vse32.v v0, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfmacc.vf v2, %0, v20" ::"f"(t1));
  asm volatile("vse32.v v2, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfmacc.vf v4, %0, v20" ::"f"(t2));
  asm volatile("vse32.v v4, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfmacc.vf v6, %0, v20" ::"f"(t3));
  asm volatile("vse32.v v6, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfmacc.vf v8, %0, v20" ::"f"(t4));
  asm volatile("vse32.v v8, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfmacc.vf v10, %0, v20" ::"f"(t5));
  asm volatile("vse32.v v10, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfmacc.vf v12, %0, v20" ::"f"(t6));
  asm volatile("vse32.v v12, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfmacc.vf v14, %0, v20" ::"f"(t7));
  asm volatile("vse32.v v14, (%0);" ::"r"(c));
}

void fmatmul_vec_8x8_tn_32(float *c, const float *a, const float *b,
                        const unsigned long int N, const unsigned long int P,
                        const unsigned long int lda) {
  float t0, t1, t2, t3, t4, t5, t6, t7;

  if (N <= 2) {
    const float *a_row = a;
    for (unsigned long int m = 0; m < N; m++) {
      asm volatile("vle32.v v18, (%0);" ::"r"(b));
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

  asm volatile("vle32.v v18, (%0);" ::"r"(b));
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

    asm volatile("vle32.v v20, (%0);" ::"r"(b));
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

    asm volatile("vle32.v v18, (%0);" ::"r"(b));
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
  asm volatile("vse32.v v0, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfmacc.vf v2, %0, v20" ::"f"(t1));
  asm volatile("vse32.v v2, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfmacc.vf v4, %0, v20" ::"f"(t2));
  asm volatile("vse32.v v4, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfmacc.vf v6, %0, v20" ::"f"(t3));
  asm volatile("vse32.v v6, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfmacc.vf v8, %0, v20" ::"f"(t4));
  asm volatile("vse32.v v8, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfmacc.vf v10, %0, v20" ::"f"(t5));
  asm volatile("vse32.v v10, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfmacc.vf v12, %0, v20" ::"f"(t6));
  asm volatile("vse32.v v12, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfmacc.vf v14, %0, v20" ::"f"(t7));
  asm volatile("vse32.v v14, (%0);" ::"r"(c));
}

// ---------------
// 16x16
// ---------------

void fmatmul_16x16_32(float *c, const float *a, const float *b,
                   unsigned long int M, unsigned long int N,
                   unsigned long int P) {
  // We work on 4 rows of the matrix at once
  const unsigned long int block_size = 16;
  unsigned long int block_size_p;

  // Set the vector configuration
  asm volatile("vsetvli %0, %1, e32, m1, ta, ma" : "=r"(block_size_p) : "r"(P));

  // Slice the matrix into a manageable number of columns p_
  for (unsigned long int p = 0; p < P; p += block_size_p) {
    // Set the vector length
    const unsigned long int p_ = MIN(P - p, block_size_p);

    // Find pointers to the submatrices
    const float *b_ = b + p;
    float *c_ = c + p;

    asm volatile("vsetvli zero, %0, e32, m1, ta, ma" ::"r"(p_));

    // Iterate over the rows
    for (unsigned long int m = 0; m < M; m += block_size) {
      // Find pointer to the submatrices
      const float *a_ = a + m * N;
      float *c__ = c_ + m * P;

      fmatmul_vec_16x16_slice_init_32();
      fmatmul_vec_16x16_32(c__, a_, b_, N, P);
    }
  }
}

void fmatmul_16x16_fused_32(float *c, const float *a, const float *b,
                         const float *bias, unsigned long int M,
                         unsigned long int N, unsigned long int P) {
  // We work on 4 rows of the matrix at once
  const unsigned long int block_size = 16;
  unsigned long int block_size_p;

  // Set the vector configuration
  asm volatile("vsetvli %0, %1, e32, m1, ta, ma" : "=r"(block_size_p) : "r"(P));

  // Slice the matrix into a manageable number of columns p_
  for (unsigned long int p = 0; p < P; p += block_size_p) {
    // Set the vector length
    const unsigned long int p_ = MIN(P - p, block_size_p);

    // Find pointers to the submatrices
    const float *b_ = b + p;
    float *c_ = c + p;
    const float *bias_slice = bias + p;

    asm volatile("vsetvli zero, %0, e32, m1, ta, ma" : : "r"(p_));

    // Iterate over the rows
    for (unsigned long int m = 0; m < M; m += block_size) {
      // Find pointer to the submatrices
      const float *a_ = a + m * N;
      float *c__ = c_ + m * P;

      fmatmul_vec_16x16_slice_init_fused_32(bias_slice);
      fmatmul_vec_16x16_32(c__, a_, b_, N, P);
    }
  }
}

void fmatmul_16x16_nt_32(float *c, const float *a, const float *b,
                      unsigned long int M, unsigned long int N,
                      unsigned long int P) {
  // We work on 4 rows of the matrix at once
  const unsigned long int block_size = 16;
  unsigned long int block_size_p;

  // Set the vector configuration
  asm volatile("vsetvli %0, %1, e32, m1, ta, ma" : "=r"(block_size_p) : "r"(P));

  // Slice the matrix into a manageable number of columns p_
  for (unsigned long int p = 0; p < P; p += block_size_p) {
    // Set the vector length
    const unsigned long int p_ = MIN(P - p, block_size_p);

    // Find pointers to the submatrices
    const float *b_ = b + p * N;
    float *c_ = c + p;

    asm volatile("vsetvli zero, %0, e32, m1, ta, ma" : : "r"(p_));

    // Iterate over the rows
    for (unsigned long int m = 0; m < M; m += block_size) {
      // Find pointer to the submatrices
      const float *a_ = a + m * N;
      float *c__ = c_ + m * P;

      fmatmul_vec_16x16_slice_init_32();
      fmatmul_vec_16x16_nt_32(c__, a_, b_, N, P);
    }
  }
}

void fmatmul_16x16_tn_32(float *c, const float *a, const float *b,
                      unsigned long int M, unsigned long int N,
                      unsigned long int P) {
  const unsigned long int block_size = 16;
  unsigned long int block_size_p;

  asm volatile("vsetvli %0, %1, e32, m1, ta, ma" : "=r"(block_size_p) : "r"(P));

  for (unsigned long int p = 0; p < P; p += block_size_p) {
    const unsigned long int p_ = MIN(P - p, block_size_p);

    const float *b_ = b + p;
    float *c_ = c + p;

    asm volatile("vsetvli zero, %0, e32, m1, ta, ma" : : "r"(p_));

    for (unsigned long int n = 0; n < N; n += block_size) {
      const float *a_ = a + n;
      float *c__ = c_ + n * P;

      fmatmul_vec_16x16_slice_init_32();
      fmatmul_vec_16x16_tn_32(c__, a_, b_, M, P, N);
    }
  }
}

void fmatmul_vec_16x16_slice_init_32() {
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

void fmatmul_vec_16x16_slice_load_bias_32(const float *bias_slice) {
  asm volatile("vle32.v v0, (%0);" ::"r"(bias_slice));
  asm volatile("vmv.v.v v1,  v0");
  asm volatile("vmv.v.v v2,  v0");
  asm volatile("vmv.v.v v3,  v0");
  asm volatile("vmv.v.v v4,  v0");
  asm volatile("vmv.v.v v5,  v0");
  asm volatile("vmv.v.v v6,  v0");
  asm volatile("vmv.v.v v7,  v0");
  asm volatile("vmv.v.v v8,  v0");
  asm volatile("vmv.v.v v9,  v0");
  asm volatile("vmv.v.v v10, v0");
  asm volatile("vmv.v.v v11, v0");
  asm volatile("vmv.v.v v12, v0");
  asm volatile("vmv.v.v v13, v0");
  asm volatile("vmv.v.v v14, v0");
  asm volatile("vmv.v.v v15, v0");
}

void fmatmul_vec_16x16_slice_init_fused_32(const float *bias_slice) {
  asm volatile("vle32.v v0, (%0);" ::"r"(bias_slice));
  asm volatile("vmv.v.v v1,  v0");
  asm volatile("vmv.v.v v2,  v0");
  asm volatile("vmv.v.v v3,  v0");
  asm volatile("vmv.v.v v4,  v0");
  asm volatile("vmv.v.v v5,  v0");
  asm volatile("vmv.v.v v6,  v0");
  asm volatile("vmv.v.v v7,  v0");
  asm volatile("vmv.v.v v8,  v0");
  asm volatile("vmv.v.v v9,  v0");
  asm volatile("vmv.v.v v10, v0");
  asm volatile("vmv.v.v v11, v0");
  asm volatile("vmv.v.v v12, v0");
  asm volatile("vmv.v.v v13, v0");
  asm volatile("vmv.v.v v14, v0");
  asm volatile("vmv.v.v v15, v0");
}

void fmatmul_vec_16x16_32(float *c, const float *a, const float *b,
                       const unsigned long int N, const unsigned long int P) {
  // Temporary variables
  float t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15;

  // Original pointer
  const float *a_ = a;

  // Prefetch one row of scalar values
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

  // Prefetch one row of matrix B
  asm volatile("vle32.v v16, (%0);" ::"r"(b));
  b += P;

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

    asm volatile("vfmacc.vf v0, %0, v16" ::"f"(t0));
    t0 = *a, a += N;

    // Load one row of B
    asm volatile("vle32.v v17, (%0);" ::"r"(b));
    b += P;

    asm volatile("vfmacc.vf v1, %0, v16" ::"f"(t1));
    t1 = *a, a += N;
    asm volatile("vfmacc.vf v2, %0, v16" ::"f"(t2));
    t2 = *a, a += N;
    asm volatile("vfmacc.vf v3, %0, v16" ::"f"(t3));
    t3 = *a, a += N;
    asm volatile("vfmacc.vf v4, %0, v16" ::"f"(t4));
    t4 = *a, a += N;
    asm volatile("vfmacc.vf v5, %0, v16" ::"f"(t5));
    t5 = *a, a += N;
    asm volatile("vfmacc.vf v6, %0, v16" ::"f"(t6));
    t6 = *a, a += N;
    asm volatile("vfmacc.vf v7, %0, v16" ::"f"(t7));
    t7 = *a, a += N;
    asm volatile("vfmacc.vf v8, %0, v16" ::"f"(t8));
    t8 = *a, a += N;
    asm volatile("vfmacc.vf v9, %0, v16" ::"f"(t9));
    t9 = *a, a += N;
    asm volatile("vfmacc.vf v10, %0, v16" ::"f"(t10));
    t10 = *a, a += N;
    asm volatile("vfmacc.vf v11, %0, v16" ::"f"(t11));
    t11 = *a, a += N;
    asm volatile("vfmacc.vf v12, %0, v16" ::"f"(t12));
    t12 = *a, a += N;
    asm volatile("vfmacc.vf v13, %0, v16" ::"f"(t13));
    t13 = *a, a += N;
    asm volatile("vfmacc.vf v14, %0, v16" ::"f"(t14));
    t14 = *a, a += N;
    asm volatile("vfmacc.vf v15, %0, v16" ::"f"(t15));
    t15 = *a;

    a = a_ + ++n;

    if (n == N)
      break;

    asm volatile("vfmacc.vf v0, %0, v17" ::"f"(t0));
    t0 = *a, a += N;

    // Load one row of B
    asm volatile("vle32.v v16, (%0);" ::"r"(b));
    b += P;

    asm volatile("vfmacc.vf v1, %0, v17" ::"f"(t1));
    t1 = *a, a += N;
    asm volatile("vfmacc.vf v2, %0, v17" ::"f"(t2));
    t2 = *a, a += N;
    asm volatile("vfmacc.vf v3, %0, v17" ::"f"(t3));
    t3 = *a, a += N;
    asm volatile("vfmacc.vf v4, %0, v17" ::"f"(t4));
    t4 = *a, a += N;
    asm volatile("vfmacc.vf v5, %0, v17" ::"f"(t5));
    t5 = *a, a += N;
    asm volatile("vfmacc.vf v6, %0, v17" ::"f"(t6));
    t6 = *a, a += N;
    asm volatile("vfmacc.vf v7, %0, v17" ::"f"(t7));
    t7 = *a, a += N;
    asm volatile("vfmacc.vf v8, %0, v17" ::"f"(t8));
    t8 = *a, a += N;
    asm volatile("vfmacc.vf v9, %0, v17" ::"f"(t9));
    t9 = *a, a += N;
    asm volatile("vfmacc.vf v10, %0, v17" ::"f"(t10));
    t10 = *a, a += N;
    asm volatile("vfmacc.vf v11, %0, v17" ::"f"(t11));
    t11 = *a, a += N;
    asm volatile("vfmacc.vf v12, %0, v17" ::"f"(t12));
    t12 = *a, a += N;
    asm volatile("vfmacc.vf v13, %0, v17" ::"f"(t13));
    t13 = *a, a += N;
    asm volatile("vfmacc.vf v14, %0, v17" ::"f"(t14));
    t14 = *a, a += N;
    asm volatile("vfmacc.vf v15, %0, v17" ::"f"(t15));
    t15 = *a;
  }

  // Last iteration: store results
  asm volatile("vfmacc.vf v0, %0, v17" ::"f"(t0));
  asm volatile("vse32.v v0, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfmacc.vf v1, %0, v17" ::"f"(t1));
  asm volatile("vse32.v v1, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfmacc.vf v2, %0, v17" ::"f"(t2));
  asm volatile("vse32.v v2, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfmacc.vf v3, %0, v17" ::"f"(t3));
  asm volatile("vse32.v v3, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfmacc.vf v4, %0, v17" ::"f"(t4));
  asm volatile("vse32.v v4, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfmacc.vf v5, %0, v17" ::"f"(t5));
  asm volatile("vse32.v v5, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfmacc.vf v6, %0, v17" ::"f"(t6));
  asm volatile("vse32.v v6, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfmacc.vf v7, %0, v17" ::"f"(t7));
  asm volatile("vse32.v v7, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfmacc.vf v8, %0, v17" ::"f"(t8));
  asm volatile("vse32.v v8, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfmacc.vf v9, %0, v17" ::"f"(t9));
  asm volatile("vse32.v v9, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfmacc.vf v10, %0, v17" ::"f"(t10));
  asm volatile("vse32.v v10, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfmacc.vf v11, %0, v17" ::"f"(t11));
  asm volatile("vse32.v v11, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfmacc.vf v12, %0, v17" ::"f"(t12));
  asm volatile("vse32.v v12, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfmacc.vf v13, %0, v17" ::"f"(t13));
  asm volatile("vse32.v v13, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfmacc.vf v14, %0, v17" ::"f"(t14));
  asm volatile("vse32.v v14, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfmacc.vf v15, %0, v17" ::"f"(t15));
  asm volatile("vse32.v v15, (%0);" ::"r"(c));
}

void fmatmul_vec_16x16_nt_32(float *c, const float *a, const float *b,
                          const unsigned long int N, const unsigned long int P) {
  // Temporary variables
  float t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15;

  // Original pointer
  const float *a_ = a;
  long stride_b_bytes = (long)N * (long)sizeof(float);

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
  asm volatile("vlse32.v v16, (%0), %1" : : "r"(b), "r"(stride_b_bytes));
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

    asm volatile("vfmacc.vf v0, %0, v16" ::"f"(t0));
    t0 = *a, a += N;

    // Load next column of B
    asm volatile("vlse32.v v17, (%0), %1" : : "r"(b), "r"(stride_b_bytes));
    b += 1;

    asm volatile("vfmacc.vf v1, %0, v16" ::"f"(t1));
    t1 = *a, a += N;
    asm volatile("vfmacc.vf v2, %0, v16" ::"f"(t2));
    t2 = *a, a += N;
    asm volatile("vfmacc.vf v3, %0, v16" ::"f"(t3));
    t3 = *a, a += N;
    asm volatile("vfmacc.vf v4, %0, v16" ::"f"(t4));
    t4 = *a, a += N;
    asm volatile("vfmacc.vf v5, %0, v16" ::"f"(t5));
    t5 = *a, a += N;
    asm volatile("vfmacc.vf v6, %0, v16" ::"f"(t6));
    t6 = *a, a += N;
    asm volatile("vfmacc.vf v7, %0, v16" ::"f"(t7));
    t7 = *a, a += N;
    asm volatile("vfmacc.vf v8, %0, v16" ::"f"(t8));
    t8 = *a, a += N;
    asm volatile("vfmacc.vf v9, %0, v16" ::"f"(t9));
    t9 = *a, a += N;
    asm volatile("vfmacc.vf v10, %0, v16" ::"f"(t10));
    t10 = *a, a += N;
    asm volatile("vfmacc.vf v11, %0, v16" ::"f"(t11));
    t11 = *a, a += N;
    asm volatile("vfmacc.vf v12, %0, v16" ::"f"(t12));
    t12 = *a, a += N;
    asm volatile("vfmacc.vf v13, %0, v16" ::"f"(t13));
    t13 = *a, a += N;
    asm volatile("vfmacc.vf v14, %0, v16" ::"f"(t14));
    t14 = *a, a += N;
    asm volatile("vfmacc.vf v15, %0, v16" ::"f"(t15));
    t15 = *a;

    a = a_ + ++n;

    if (n == N)
      break;

    asm volatile("vfmacc.vf v0, %0, v17" ::"f"(t0));
    t0 = *a, a += N;

    // Load next column of B
    asm volatile("vlse32.v v16, (%0), %1" : : "r"(b), "r"(stride_b_bytes));
    b += 1;

    asm volatile("vfmacc.vf v1, %0, v17" ::"f"(t1));
    t1 = *a, a += N;
    asm volatile("vfmacc.vf v2, %0, v17" ::"f"(t2));
    t2 = *a, a += N;
    asm volatile("vfmacc.vf v3, %0, v17" ::"f"(t3));
    t3 = *a, a += N;
    asm volatile("vfmacc.vf v4, %0, v17" ::"f"(t4));
    t4 = *a, a += N;
    asm volatile("vfmacc.vf v5, %0, v17" ::"f"(t5));
    t5 = *a, a += N;
    asm volatile("vfmacc.vf v6, %0, v17" ::"f"(t6));
    t6 = *a, a += N;
    asm volatile("vfmacc.vf v7, %0, v17" ::"f"(t7));
    t7 = *a, a += N;
    asm volatile("vfmacc.vf v8, %0, v17" ::"f"(t8));
    t8 = *a, a += N;
    asm volatile("vfmacc.vf v9, %0, v17" ::"f"(t9));
    t9 = *a, a += N;
    asm volatile("vfmacc.vf v10, %0, v17" ::"f"(t10));
    t10 = *a, a += N;
    asm volatile("vfmacc.vf v11, %0, v17" ::"f"(t11));
    t11 = *a, a += N;
    asm volatile("vfmacc.vf v12, %0, v17" ::"f"(t12));
    t12 = *a, a += N;
    asm volatile("vfmacc.vf v13, %0, v17" ::"f"(t13));
    t13 = *a, a += N;
    asm volatile("vfmacc.vf v14, %0, v17" ::"f"(t14));
    t14 = *a, a += N;
    asm volatile("vfmacc.vf v15, %0, v17" ::"f"(t15));
    t15 = *a;
  }

  // Last iteration: store results
  asm volatile("vfmacc.vf v0, %0, v17" ::"f"(t0));
  asm volatile("vse32.v v0, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfmacc.vf v1, %0, v17" ::"f"(t1));
  asm volatile("vse32.v v1, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfmacc.vf v2, %0, v17" ::"f"(t2));
  asm volatile("vse32.v v2, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfmacc.vf v3, %0, v17" ::"f"(t3));
  asm volatile("vse32.v v3, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfmacc.vf v4, %0, v17" ::"f"(t4));
  asm volatile("vse32.v v4, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfmacc.vf v5, %0, v17" ::"f"(t5));
  asm volatile("vse32.v v5, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfmacc.vf v6, %0, v17" ::"f"(t6));
  asm volatile("vse32.v v6, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfmacc.vf v7, %0, v17" ::"f"(t7));
  asm volatile("vse32.v v7, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfmacc.vf v8, %0, v17" ::"f"(t8));
  asm volatile("vse32.v v8, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfmacc.vf v9, %0, v17" ::"f"(t9));
  asm volatile("vse32.v v9, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfmacc.vf v10, %0, v17" ::"f"(t10));
  asm volatile("vse32.v v10, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfmacc.vf v11, %0, v17" ::"f"(t11));
  asm volatile("vse32.v v11, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfmacc.vf v12, %0, v17" ::"f"(t12));
  asm volatile("vse32.v v12, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfmacc.vf v13, %0, v17" ::"f"(t13));
  asm volatile("vse32.v v13, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfmacc.vf v14, %0, v17" ::"f"(t14));
  asm volatile("vse32.v v14, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfmacc.vf v15, %0, v17" ::"f"(t15));
  asm volatile("vse32.v v15, (%0);" ::"r"(c));
}

void fmatmul_vec_16x16_tn_32(float *c, const float *a, const float *b,
                          const unsigned long int N, const unsigned long int P,
                          const unsigned long int lda) {
  float t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15;

  if (N <= 2) {
    const float *a_row = a;
    for (unsigned long int m = 0; m < N; m++) {
      asm volatile("vle32.v v16, (%0);" ::"r"(b));
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

  asm volatile("vle32.v v16, (%0);" ::"r"(b));
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

    asm volatile("vle32.v v17, (%0);" ::"r"(b));
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

    asm volatile("vle32.v v16, (%0);" ::"r"(b));
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
  asm volatile("vse32.v v0, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfmacc.vf v1, %0, v17" ::"f"(t1));
  asm volatile("vse32.v v1, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfmacc.vf v2, %0, v17" ::"f"(t2));
  asm volatile("vse32.v v2, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfmacc.vf v3, %0, v17" ::"f"(t3));
  asm volatile("vse32.v v3, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfmacc.vf v4, %0, v17" ::"f"(t4));
  asm volatile("vse32.v v4, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfmacc.vf v5, %0, v17" ::"f"(t5));
  asm volatile("vse32.v v5, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfmacc.vf v6, %0, v17" ::"f"(t6));
  asm volatile("vse32.v v6, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfmacc.vf v7, %0, v17" ::"f"(t7));
  asm volatile("vse32.v v7, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfmacc.vf v8, %0, v17" ::"f"(t8));
  asm volatile("vse32.v v8, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfmacc.vf v9, %0, v17" ::"f"(t9));
  asm volatile("vse32.v v9, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfmacc.vf v10, %0, v17" ::"f"(t10));
  asm volatile("vse32.v v10, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfmacc.vf v11, %0, v17" ::"f"(t11));
  asm volatile("vse32.v v11, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfmacc.vf v12, %0, v17" ::"f"(t12));
  asm volatile("vse32.v v12, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfmacc.vf v13, %0, v17" ::"f"(t13));
  asm volatile("vse32.v v13, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfmacc.vf v14, %0, v17" ::"f"(t14));
  asm volatile("vse32.v v14, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfmacc.vf v15, %0, v17" ::"f"(t15));
  asm volatile("vse32.v v15, (%0);" ::"r"(c));
}
