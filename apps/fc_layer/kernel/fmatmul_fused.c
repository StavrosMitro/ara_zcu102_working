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

void fmatmul_fused(float *c, const _Float16 *a, const _Float16 *b,
                   const float *bias, const unsigned long int M,
                   const unsigned long int N, const unsigned long int P) {
  if (M <= 4) {
    fmatmul_4x4_fused(c, a, b, bias, M, N, P);
  } else if (M <= 8) {
    fmatmul_8x8_fused(c, a, b, bias, M, N, P);
  } else if (M <= 64) {
    fmatmul_16x16_fused(c, a, b, bias, M, N, P);
  } else if (M <= 128) {
    // With an 8x8 matmul, use LMUL=2 to increase vl.
    fmatmul_8x8_fused(c, a, b, bias, M, N, P);
  } else {
    // With a 4x4 matmul, use LMUL=4 to increase vl.
    fmatmul_4x4_fused(c, a, b, bias, M, N, P);
  }
}

void fmatmul_4x4_fused(float *c, const _Float16 *a, const _Float16 *b,
                       const float *bias, const unsigned long int M,
                       const unsigned long int N, const unsigned long int P) {
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
    const _Float16 *b_ = b + p;
    float *c_ = c + p;
    const float *bias_slice = bias + p;

    asm volatile("vsetvli zero, %0, e16, m2, ta, ma" : : "r"(p_));

    // Iterate over the rows
    for (unsigned long int m = 0; m < M; m += block_size) {
      // Find pointer to the submatrices
      const _Float16 *a_ = a + m * N;
      float *c__ = c_ + m * P;

      fmatmul_vec_4x4_slice_init_fused(bias_slice, p_);
      fmatmul_vec_4x4(c__, a_, b_, N, P, p_);
    }
  }
}

void fmatmul_vec_4x4_slice_load_bias(const float *bias_slice, unsigned long int vl) {
  asm volatile("vsetvli zero, %0, e32, m4, ta, ma" :: "r"(vl));
  asm volatile("vle32.v v0, (%0);" ::"r"(bias_slice));
  asm volatile("vmv.v.v v4,  v0");
  asm volatile("vmv.v.v v8,  v0");
  asm volatile("vmv.v.v v12, v0");
  asm volatile("vsetvli zero, %0, e16, m2, ta, ma" :: "r"(vl));
}

void fmatmul_vec_4x4_slice_init_fused(const float *bias_slice, unsigned long int vl) {
  asm volatile("vsetvli zero, %0, e32, m4, ta, ma" :: "r"(vl));
  asm volatile("vle32.v v0, (%0);" ::"r"(bias_slice));
  asm volatile("vmv.v.v v4,  v0");
  asm volatile("vmv.v.v v8,  v0");
  asm volatile("vmv.v.v v12, v0");
  asm volatile("vsetvli zero, %0, e16, m2, ta, ma" :: "r"(vl));
}

void fmatmul_8x8_fused(float *c, const _Float16 *a, const _Float16 *b,
                       const float *bias, const unsigned long int M,
                       const unsigned long int N, const unsigned long int P) {
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
    const _Float16 *b_ = b + p;
    float *c_ = c + p;
    const float *bias_slice = bias + p;

    asm volatile("vsetvli zero, %0, e16, m1, ta, ma" : : "r"(p_));

    // Iterate over the rows
    for (unsigned long int m = 0; m < M; m += block_size) {
      // Find pointer to the submatrices
      const _Float16 *a_ = a + m * N;
      float *c__ = c_ + m * P;

      fmatmul_vec_8x8_slice_init_fused(bias_slice, p_);
      fmatmul_vec_8x8(c__, a_, b_, N, P, p_);
    }
  }
}

void fmatmul_vec_8x8_slice_load_bias(const float *bias_slice, unsigned long int vl) {
  asm volatile("vsetvli zero, %0, e32, m2, ta, ma" :: "r"(vl));
  asm volatile("vle32.v v0, (%0);" ::"r"(bias_slice));
  asm volatile("vmv.v.v v2,  v0");
  asm volatile("vmv.v.v v4,  v0");
  asm volatile("vmv.v.v v6,  v0");
  asm volatile("vmv.v.v v8,  v0");
  asm volatile("vmv.v.v v10, v0");
  asm volatile("vmv.v.v v12, v0");
  asm volatile("vmv.v.v v14, v0");
  asm volatile("vsetvli zero, %0, e16, m1, ta, ma" :: "r"(vl));
}

void fmatmul_vec_8x8_slice_init_fused(const float *bias_slice, unsigned long int vl) {
  asm volatile("vsetvli zero, %0, e32, m2, ta, ma" :: "r"(vl));
  asm volatile("vle32.v v0, (%0);" ::"r"(bias_slice));
  asm volatile("vmv.v.v v2,  v0");
  asm volatile("vmv.v.v v4,  v0");
  asm volatile("vmv.v.v v6,  v0");
  asm volatile("vmv.v.v v8,  v0");
  asm volatile("vmv.v.v v10, v0");
  asm volatile("vmv.v.v v12, v0");
  asm volatile("vmv.v.v v14, v0");
  asm volatile("vsetvli zero, %0, e16, m1, ta, ma" :: "r"(vl));
}

void fmatmul_16x16_fused(float *c, const _Float16 *a, const _Float16 *b,
                         const float *bias, unsigned long int M,
                         unsigned long int N, unsigned long int P) {
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
    const _Float16 *b_ = b + p;
    float *c_ = c + p;
    const float *bias_slice = bias + p;

    asm volatile("vsetvli zero, %0, e16, mf2, ta, ma" : : "r"(p_));

    // Iterate over the rows
    for (unsigned long int m = 0; m < M; m += block_size) {
      // Find pointer to the submatrices
      const _Float16 *a_ = a + m * N;
      float *c__ = c_ + m * P;

      fmatmul_vec_16x16_slice_init_fused(bias_slice, p_);
      fmatmul_vec_16x16(c__, a_, b_, N, P, p_);
    }
  }
}

void fmatmul_vec_16x16_slice_load_bias(const float *bias_slice, unsigned long int vl) {
  asm volatile("vsetvli zero, %0, e32, m1, ta, ma" :: "r"(vl));
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
  asm volatile("vsetvli zero, %0, e16, mf2, ta, ma" :: "r"(vl));
}

void fmatmul_vec_16x16_slice_init_fused(const float *bias_slice, unsigned long int vl) {
  asm volatile("vsetvli zero, %0, e32, m1, ta, ma" :: "r"(vl));
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
  asm volatile("vsetvli zero, %0, e16, mf2, ta, ma" :: "r"(vl));
}

