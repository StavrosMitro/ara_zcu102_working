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
// Dispatch: select tile size by M (Fused version)
// ---------------------------------------------------------------------------
void fmatmul_fused_16(_Float16 *c, const _Float16 *a, const _Float16 *b,
                      const _Float16 *bias, const unsigned long int M,
                      const unsigned long int N, const unsigned long int P) {
    if (M <= 4) {
        fmatmul_4x4_fused_16(c, a, b, bias, M, N, P);
    } else if (M <= 8) {
        fmatmul_8x8_fused_16(c, a, b, bias, M, N, P);
    } else if (M <= 64) {
        fmatmul_16x16_fused_16(c, a, b, bias, M, N, P);
    } else if (M <= 128) {
        fmatmul_8x8_fused_16(c, a, b, bias, M, N, P);
    } else {
        fmatmul_4x4_fused_16(c, a, b, bias, M, N, P);
    }
}

// ---------------
// 4x4 Fused
// ---------------

void fmatmul_4x4_fused_16(_Float16 *c, const _Float16 *a, const _Float16 *b,
                           const _Float16 *bias, const unsigned long int M,
                           const unsigned long int N, const unsigned long int P) {
    const unsigned long int block_size = 4;
    unsigned long int block_size_p;

    asm volatile("vsetvli %0, %1, e16, m4, ta, ma" : "=r"(block_size_p) : "r"(P));

    for (unsigned long int p = 0; p < P; p += block_size_p) {
        const unsigned long int p_ = MIN(P - p, block_size_p);

        const _Float16 *b_ = b + p;
        _Float16 *c_ = c + p;
        const _Float16 *bias_slice = bias + p;

        asm volatile("vsetvli zero, %0, e16, m4, ta, ma" :: "r"(p_));

        for (unsigned long int m = 0; m < M; m += block_size) {
            const _Float16 *a_ = a + m * N;
            _Float16 *c__ = c_ + m * P;

            fmatmul_vec_4x4_slice_init_fused_16(bias_slice);
            fmatmul_vec_4x4_16(c__, a_, b_, N, P);
        }
    }
}

void fmatmul_vec_4x4_slice_init_fused_16(const _Float16 *bias_slice) {
    asm volatile("vle16.v v0, (%0);" :: "r"(bias_slice));
    asm volatile("vmv.v.v v4,  v0");
    asm volatile("vmv.v.v v8,  v0");
    asm volatile("vmv.v.v v12, v0");
}


// ---------------
// 8x8 Fused
// ---------------

void fmatmul_8x8_fused_16(_Float16 *c, const _Float16 *a, const _Float16 *b,
                           const _Float16 *bias, const unsigned long int M,
                           const unsigned long int N, const unsigned long int P) {
    const unsigned long int block_size = 8;
    unsigned long int block_size_p;

    asm volatile("vsetvli %0, %1, e16, m2, ta, ma" : "=r"(block_size_p) : "r"(P));

    for (unsigned long int p = 0; p < P; p += block_size_p) {
        const unsigned long int p_ = MIN(P - p, block_size_p);

        const _Float16 *b_ = b + p;
        _Float16 *c_ = c + p;
        const _Float16 *bias_slice = bias + p;

        asm volatile("vsetvli zero, %0, e16, m2, ta, ma" :: "r"(p_));

        for (unsigned long int m = 0; m < M; m += block_size) {
            const _Float16 *a_ = a + m * N;
            _Float16 *c__ = c_ + m * P;

            fmatmul_vec_8x8_slice_init_fused_16(bias_slice);
            fmatmul_vec_8x8_16(c__, a_, b_, N, P);
        }
    }
}

void fmatmul_vec_8x8_slice_init_fused_16(const _Float16 *bias_slice) {
    asm volatile("vle16.v v0, (%0);" :: "r"(bias_slice));
    asm volatile("vmv.v.v v2,  v0");
    asm volatile("vmv.v.v v4,  v0");
    asm volatile("vmv.v.v v6,  v0");
    asm volatile("vmv.v.v v8,  v0");
    asm volatile("vmv.v.v v10, v0");
    asm volatile("vmv.v.v v12, v0");
    asm volatile("vmv.v.v v14, v0");
}


// ---------------
// 16x16 Fused
// ---------------

void fmatmul_16x16_fused_16(_Float16 *c, const _Float16 *a, const _Float16 *b,
                             const _Float16 *bias, unsigned long int M,
                             unsigned long int N, unsigned long int P) {
    const unsigned long int block_size = 16;
    unsigned long int block_size_p;

    asm volatile("vsetvli %0, %1, e16, m1, ta, ma" : "=r"(block_size_p) : "r"(P));

    for (unsigned long int p = 0; p < P; p += block_size_p) {
        const unsigned long int p_ = MIN(P - p, block_size_p);

        const _Float16 *b_ = b + p;
        _Float16 *c_ = c + p;
        const _Float16 *bias_slice = bias + p;

        asm volatile("vsetvli zero, %0, e16, m1, ta, ma" :: "r"(p_));

        for (unsigned long int m = 0; m < M; m += block_size) {
            const _Float16 *a_ = a + m * N;
            _Float16 *c__ = c_ + m * P;

            fmatmul_vec_16x16_slice_init_fused_16(bias_slice);
            fmatmul_vec_16x16_16(c__, a_, b_, N, P);
        }
    }
}

void fmatmul_vec_16x16_slice_init_fused_16(const _Float16 *bias_slice) {
    asm volatile("vle16.v v0, (%0);" :: "r"(bias_slice));
    asm volatile("vmv.v.v v1,  v0"); asm volatile("vmv.v.v v2,  v0");
    asm volatile("vmv.v.v v3,  v0"); asm volatile("vmv.v.v v4,  v0");
    asm volatile("vmv.v.v v5,  v0"); asm volatile("vmv.v.v v6,  v0");
    asm volatile("vmv.v.v v7,  v0"); asm volatile("vmv.v.v v8,  v0");
    asm volatile("vmv.v.v v9,  v0"); asm volatile("vmv.v.v v10, v0");
    asm volatile("vmv.v.v v11, v0"); asm volatile("vmv.v.v v12, v0");
    asm volatile("vmv.v.v v13, v0"); asm volatile("vmv.v.v v14, v0");
    asm volatile("vmv.v.v v15, v0");
}


// NOTE: fmatmul_vec_4x4_16 / _8x8_16 / _16x16_16 are NOT defined here. The fused
// variants differ from the plain ones only in slice_init (bias-seeded accumulator
// vs zeroed); the inner compute kernels are identical, so they live once in
// fmatmul.c and are declared in fmatmul.h. Byte-identical copies used to sit here
// and collided at link time.
