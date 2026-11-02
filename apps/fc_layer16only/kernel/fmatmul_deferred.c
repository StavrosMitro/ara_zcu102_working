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

// ---------------------------------------------------------------------------
// Deferred variant (4x4 only, A * B^T)
// ---------------------------------------------------------------------------

void fmatmul_deferred_16(_Float16 *c, const _Float16 *a, const _Float16 *b,
                         const unsigned long int M, const unsigned long int N,
                         const unsigned long int P) {
    fmatmul_4x4_deferred_16(c, a, b, M, N, P);
}

void fmatmul_4x4_deferred_16(_Float16 *c, const _Float16 *a, const _Float16 *b,
                              const unsigned long int M, const unsigned long int N,
                              const unsigned long int P) {
    const unsigned long int block_size = 4;

    for (unsigned long int p = 0; p < P; p += block_size) {
        const unsigned long int p_ = MIN(P - p, block_size);

        const _Float16 *b_ = b + p * N;
        _Float16 *c_ = c + p;

        unsigned long int block_size_n;
        asm volatile("vsetvli %0, %1, e16, m4, ta, ma" : "=r"(block_size_n) : "r"(N));

        for (unsigned long int m = 0; m < M; m += block_size) {
            const _Float16 *a_ = a + m * N;
            _Float16 *c__ = c_ + m * P;

            fmatmul_vec_4x4_slice_init_deferred_16();
            fmatmul_vec_4x4_deferred_16(c__, a_, b_, N, P, p_);
        }
    }
}

void fmatmul_vec_4x4_slice_init_deferred_16(void) {
    asm volatile("vmv.v.i v0,  0");
    asm volatile("vmv.v.i v4,  0");
    asm volatile("vmv.v.i v8,  0");
    asm volatile("vmv.v.i v12, 0");
}

void fmatmul_vec_4x4_deferred_16(_Float16 *c, const _Float16 *a, const _Float16 *b,
                                  const unsigned long int N, const unsigned long int P,
                                  const unsigned long int p_) {
    unsigned long int block_size_n;
    asm volatile("vsetvli %0, %1, e16, m4, ta, ma" : "=r"(block_size_n) : "r"(N));

    for (unsigned long int n = 0; n < N; n += block_size_n) {
        const unsigned long int n_ = MIN(N - n, block_size_n);

        asm volatile("vsetvli zero, %0, e16, m4, ta, ma" :: "r"(n_));

        asm volatile("vle16.v v16, (%0);" :: "r"(a + n));
        asm volatile("vle16.v v20, (%0);" :: "r"(b + n));

        asm volatile("vfmacc.vv v0,  v16, v20");
        asm volatile("vfmacc.vv v4,  v16, v20");
        asm volatile("vfmacc.vv v8,  v16, v20");
        asm volatile("vfmacc.vv v12, v16, v20");
    }

    asm volatile("vsetvli zero, %0, e16, m4, ta, ma" :: "r"(p_));
    asm volatile("vfredusum.vs v0, v0, v0");
    asm volatile("vfredusum.vs v4, v4, v4");
    asm volatile("vfredusum.vs v8, v8, v8");
    asm volatile("vfredusum.vs v12, v12, v12");

    _Float16 res0, res1, res2, res3;
    asm volatile("vfmv.f.s %0, v0" : "=f"(res0));
    asm volatile("vfmv.f.s %0, v4" : "=f"(res1));
    asm volatile("vfmv.f.s %0, v8" : "=f"(res2));
    asm volatile("vfmv.f.s %0, v12" : "=f"(res3));

    *c = res0; c += P;
    *c = res1; c += P;
    *c = res2; c += P;
    *c = res3;
}
