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

static inline void fmatmul_vec_4x4_deferred(float *c, const _Float16 *a,
                                            const _Float16 *b,
                                            const unsigned long int N,
                                            const unsigned long int ldc) {
  const _Float16 *a0 = a;
  const _Float16 *a1 = a + N;
  const _Float16 *a2 = a + 2 * N;
  const _Float16 *a3 = a + 3 * N;

  const _Float16 *b0 = b;
  const _Float16 *b1 = b + N;
  const _Float16 *b2 = b + 2 * N;
  const _Float16 *b3 = b + 3 * N;

  // 1. Αρχικοποίηση 16 Accumulators (v16-v31) ως FP32 (e32, m1)
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
    // 2. Χρησιμοποιούμε Fractional LMUL (mf2) για τα 16-bit inputs!
    asm volatile("vsetvli %0, %1, e16, mf2, ta, ma" : "=r"(vl) : "r"(k));

    asm volatile("vle16.v v0, (%0)" : : "r"(a0)); a0 += vl;
    asm volatile("vle16.v v1, (%0)" : : "r"(a1)); a1 += vl;
    asm volatile("vle16.v v2, (%0)" : : "r"(a2)); a2 += vl;
    asm volatile("vle16.v v3, (%0)" : : "r"(a3)); a3 += vl;

    asm volatile("vle16.v v4, (%0)" : : "r"(b0)); b0 += vl;
    asm volatile("vle16.v v5, (%0)" : : "r"(b1)); b1 += vl;
    asm volatile("vle16.v v6, (%0)" : : "r"(b2)); b2 += vl;
    asm volatile("vle16.v v7, (%0)" : : "r"(b3)); b3 += vl;

    // Τα dest (v16-v31) γίνονται m1 με ασφάλεια
    asm volatile("vfwmacc.vv v16, v0, v4");
    asm volatile("vfwmacc.vv v17, v0, v5");
    asm volatile("vfwmacc.vv v18, v0, v6");
    asm volatile("vfwmacc.vv v19, v0, v7");

    asm volatile("vfwmacc.vv v20, v1, v4");
    asm volatile("vfwmacc.vv v21, v1, v5");
    asm volatile("vfwmacc.vv v22, v1, v6");
    asm volatile("vfwmacc.vv v23, v1, v7");

    asm volatile("vfwmacc.vv v24, v2, v4");
    asm volatile("vfwmacc.vv v25, v2, v5");
    asm volatile("vfwmacc.vv v26, v2, v6");
    asm volatile("vfwmacc.vv v27, v2, v7");

    asm volatile("vfwmacc.vv v28, v3, v4");
    asm volatile("vfwmacc.vv v29, v3, v5");
    asm volatile("vfwmacc.vv v30, v3, v6");
    asm volatile("vfwmacc.vv v31, v3, v7");

    k -= vl;
  }

  // 3. Επιστροφή σε FP32 (e32, m1) για το Reduction
  asm volatile("vsetvli zero, %0, e32, m1, ta, ma" : : "r"(N));
  asm volatile("vmv.v.i v8, 0"); // scalar 0 για το sum

  // vfredsum.vs (όχι vredsum)
  asm volatile("vfredsum.vs v16, v16, v8");
  asm volatile("vfredsum.vs v17, v17, v8");
  asm volatile("vfredsum.vs v18, v18, v8");
  asm volatile("vfredsum.vs v19, v19, v8");
  asm volatile("vfredsum.vs v20, v20, v8");
  asm volatile("vfredsum.vs v21, v21, v8");
  asm volatile("vfredsum.vs v22, v22, v8");
  asm volatile("vfredsum.vs v23, v23, v8");
  asm volatile("vfredsum.vs v24, v24, v8");
  asm volatile("vfredsum.vs v25, v25, v8");
  asm volatile("vfredsum.vs v26, v26, v8");
  asm volatile("vfredsum.vs v27, v27, v8");
  asm volatile("vfredsum.vs v28, v28, v8");
  asm volatile("vfredsum.vs v29, v29, v8");
  asm volatile("vfredsum.vs v30, v30, v8");
  asm volatile("vfredsum.vs v31, v31, v8");

  float *c_ptr = c;
  float res;

  asm volatile("vfmv.f.s %0, v16" : "=f"(res)); c_ptr[0] += res;
  asm volatile("vfmv.f.s %0, v17" : "=f"(res)); c_ptr[1] += res;
  asm volatile("vfmv.f.s %0, v18" : "=f"(res)); c_ptr[2] += res;
  asm volatile("vfmv.f.s %0, v19" : "=f"(res)); c_ptr[3] += res;
  c_ptr += ldc;

  asm volatile("vfmv.f.s %0, v20" : "=f"(res)); c_ptr[0] += res;
  asm volatile("vfmv.f.s %0, v21" : "=f"(res)); c_ptr[1] += res;
  asm volatile("vfmv.f.s %0, v22" : "=f"(res)); c_ptr[2] += res;
  asm volatile("vfmv.f.s %0, v23" : "=f"(res)); c_ptr[3] += res;
  c_ptr += ldc;

  asm volatile("vfmv.f.s %0, v24" : "=f"(res)); c_ptr[0] += res;
  asm volatile("vfmv.f.s %0, v25" : "=f"(res)); c_ptr[1] += res;
  asm volatile("vfmv.f.s %0, v26" : "=f"(res)); c_ptr[2] += res;
  asm volatile("vfmv.f.s %0, v27" : "=f"(res)); c_ptr[3] += res;
  c_ptr += ldc;

  asm volatile("vfmv.f.s %0, v28" : "=f"(res)); c_ptr[0] += res;
  asm volatile("vfmv.f.s %0, v29" : "=f"(res)); c_ptr[1] += res;
  asm volatile("vfmv.f.s %0, v30" : "=f"(res)); c_ptr[2] += res;
  asm volatile("vfmv.f.s %0, v31" : "=f"(res)); c_ptr[3] += res;
}

void fmatmul_4x4_deferred(float *c, const _Float16 *a, const _Float16 *b,
                          const unsigned long int M, const unsigned long int N,
                          const unsigned long int P) {
  // C = A * B^T
  // A: [M x N], B: [P x N], C: [M x P]
  for (unsigned long int m = 0; m < M; m += 4) {
    for (unsigned long int p = 0; p < P; p += 4) {
      const _Float16 *a_block = a + m * N;
      const _Float16 *b_block = b + p * N;
      float *c_block = c + m * P + p;

      fmatmul_vec_4x4_deferred(c_block, a_block, b_block, N, P);
    }
  }
}

