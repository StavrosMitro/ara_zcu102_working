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

#ifndef FMATMUL_H
#define FMATMUL_H

#include <stdint.h>

void fmatmul(float *c, const _Float16 *a, const _Float16 *b, unsigned long int m,
             unsigned long int n, unsigned long int p);

void fmatmul_fused(float *c, const _Float16 *a, const _Float16 *b,
                   const float *bias, unsigned long int m,
                   unsigned long int n, unsigned long int p);

// Mixed-precision variant: FP16 in/weights, FP32 widening accumulate, FP16 out.
// Used by hidden (non-terminal) fc layers so activations stay FP16.
void fmatmul_fused_f16out(_Float16 *c, const _Float16 *a, const _Float16 *b,
                          const float *bias, unsigned long int m,
                          unsigned long int n, unsigned long int p);
void fmatmul_4x4_fused_f16out(_Float16 *c, const _Float16 *a, const _Float16 *b,
                              const float *bias, unsigned long int m,
                              unsigned long int n, unsigned long int p);
void fmatmul_8x8_fused_f16out(_Float16 *c, const _Float16 *a, const _Float16 *b,
                              const float *bias, unsigned long int m,
                              unsigned long int n, unsigned long int p);
void fmatmul_16x16_fused_f16out(_Float16 *c, const _Float16 *a, const _Float16 *b,
                                const float *bias, unsigned long int m,
                                unsigned long int n, unsigned long int p);

void fmatmul_nt(float *c, const _Float16 *a, const _Float16 *b, unsigned long int m,
                unsigned long int n, unsigned long int p);

// Mixed-precision variant: FP16 in, FP32 widening accumulate, FP16 out.
// Computes backward d_input as FP16 (the layer below expects FP16 d_output).
void fmatmul_nt_f16out(_Float16 *c, const _Float16 *a, const _Float16 *b,
                       unsigned long int m, unsigned long int n, unsigned long int p);
void fmatmul_4x4_nt_f16out(_Float16 *c, const _Float16 *a, const _Float16 *b,
                           unsigned long int m, unsigned long int n, unsigned long int p);
void fmatmul_8x8_nt_f16out(_Float16 *c, const _Float16 *a, const _Float16 *b,
                           unsigned long int m, unsigned long int n, unsigned long int p);
void fmatmul_16x16_nt_f16out(_Float16 *c, const _Float16 *a, const _Float16 *b,
                             unsigned long int m, unsigned long int n, unsigned long int p);
void fmatmul_tn(float *c, const _Float16 *a, const _Float16 *b, unsigned long int m,
                unsigned long int n, unsigned long int p);

void fmatmul_4x4(float *c, const _Float16 *a, const _Float16 *b,
                 unsigned long int m, unsigned long int n, unsigned long int p);
void fmatmul_vec_4x4_slice_init(unsigned long int vl);
void fmatmul_vec_4x4_slice_load_bias(const float *bias_slice, unsigned long int vl);
void fmatmul_vec_4x4_slice_init_fused(const float *bias_slice, unsigned long int vl);
void fmatmul_vec_4x4(float *c, const _Float16 *a, const _Float16 *b,
                     unsigned long int n, unsigned long int p, unsigned long int vl);

void fmatmul_4x4_nt(float *c, const _Float16 *a, const _Float16 *b,
                    unsigned long int m, unsigned long int n,
                    unsigned long int p);
void fmatmul_vec_4x4_nt(float *c, const _Float16 *a, const _Float16 *b,
                        unsigned long int n, unsigned long int p, unsigned long int vl);
void fmatmul_vec_4x4_tn(float *c, const _Float16 *a, const _Float16 *b,
                        unsigned long int n, unsigned long int p,
                        unsigned long int lda, unsigned long int vl);
void fmatmul_4x4_tn(float *c, const _Float16 *a, const _Float16 *b,
                    unsigned long int m, unsigned long int n,
                    unsigned long int p);

void fmatmul_4x4_deferred(float *c, const _Float16 *a, const _Float16 *b,
                          unsigned long int m, unsigned long int n,
                          unsigned long int p);

void fmatmul_4x4_fused(float *c, const _Float16 *a, const _Float16 *b,
                       const float *bias, unsigned long int m,
                       unsigned long int n, unsigned long int p);

void fmatmul_8x8(float *c, const _Float16 *a, const _Float16 *b,
                 unsigned long int m, unsigned long int n, unsigned long int p);
void fmatmul_vec_8x8_slice_init(unsigned long int vl);
void fmatmul_vec_8x8_slice_load_bias(const float *bias_slice, unsigned long int vl);
void fmatmul_vec_8x8_slice_init_fused(const float *bias_slice, unsigned long int vl);
void fmatmul_vec_8x8(float *c, const _Float16 *a, const _Float16 *b,
                     unsigned long int n, unsigned long int p, unsigned long int vl);

void fmatmul_8x8_nt(float *c, const _Float16 *a, const _Float16 *b,
                    unsigned long int m, unsigned long int n,
                    unsigned long int p);
void fmatmul_vec_8x8_nt(float *c, const _Float16 *a, const _Float16 *b,
                        unsigned long int n, unsigned long int p, unsigned long int vl);
void fmatmul_8x8_tn(float *c, const _Float16 *a, const _Float16 *b,
                    unsigned long int m, unsigned long int n,
                    unsigned long int p);
void fmatmul_vec_8x8_tn(float *c, const _Float16 *a, const _Float16 *b,
                        unsigned long int n, unsigned long int p,
                        unsigned long int lda, unsigned long int vl);

void fmatmul_8x8_fused(float *c, const _Float16 *a, const _Float16 *b,
                       const float *bias, unsigned long int m,
                       unsigned long int n, unsigned long int p);

void fmatmul_16x16(float *c, const _Float16 *a, const _Float16 *b,
                   unsigned long int m, unsigned long int n,
                   unsigned long int p);
void fmatmul_vec_16x16_slice_init(unsigned long int vl);
void fmatmul_vec_16x16_slice_load_bias(const float *bias_slice, unsigned long int vl);
void fmatmul_vec_16x16_slice_init_fused(const float *bias_slice, unsigned long int vl);
void fmatmul_vec_16x16(float *c, const _Float16 *a, const _Float16 *b,
                       unsigned long int n, unsigned long int p, unsigned long int vl);

void fmatmul_16x16_nt(float *c, const _Float16 *a, const _Float16 *b,
                      unsigned long int m, unsigned long int n,
                      unsigned long int p);
void fmatmul_vec_16x16_nt(float *c, const _Float16 *a, const _Float16 *b,
                          unsigned long int n, unsigned long int p, unsigned long int vl);
void fmatmul_16x16_tn(float *c, const _Float16 *a, const _Float16 *b,
                      unsigned long int m, unsigned long int n,
                      unsigned long int p);
void fmatmul_vec_16x16_tn(float *c, const _Float16 *a, const _Float16 *b,
                          unsigned long int n, unsigned long int p,
                          unsigned long int lda, unsigned long int vl);

void fmatmul_16x16_fused(float *c, const _Float16 *a, const _Float16 *b,
                         const float *bias, unsigned long int m,
                         unsigned long int n, unsigned long int p);

void calc_bias_gradient_vec(float *d_bias, const float *d_output, int out_units, int batchsize);


#define DELTA 0.000001

extern int64_t event_trigger;

#endif
