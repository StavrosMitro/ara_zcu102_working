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

void fmatmul_32(float *c, const float *a, const float *b, unsigned long int m,
             unsigned long int n, unsigned long int p);

void fmatmul_fused_32(float *c, const float *a, const float *b,
                   const float *bias, unsigned long int m,
                   unsigned long int n, unsigned long int p);

void fmatmul_nt_32(float *c, const float *a, const float *b, unsigned long int m,
                unsigned long int n, unsigned long int p);
void fmatmul_tn_32(float *c, const float *a, const float *b, unsigned long int m,
                unsigned long int n, unsigned long int p);

void fmatmul_4x4_32(float *c, const float *a, const float *b,
                 unsigned long int m, unsigned long int n, unsigned long int p);
void fmatmul_vec_4x4_slice_init_32();
void fmatmul_vec_4x4_slice_load_bias_32(const float *bias_slice);
void fmatmul_vec_4x4_slice_init_fused_32(const float *bias_slice);
void fmatmul_vec_4x4_32(float *c, const float *a, const float *b,
                     unsigned long int n, unsigned long int p);

void fmatmul_4x4_nt_32(float *c, const float *a, const float *b,
                    unsigned long int m, unsigned long int n,
                    unsigned long int p);
void fmatmul_vec_4x4_nt_32(float *c, const float *a, const float *b,
                        unsigned long int n, unsigned long int p);
void fmatmul_vec_4x4_tn_32(float *c, const float *a, const float *b,
                        unsigned long int n, unsigned long int p,
                        unsigned long int lda);
void fmatmul_4x4_tn_32(float *c, const float *a, const float *b,
                    unsigned long int m, unsigned long int n,
                    unsigned long int p);

void fmatmul_4x4_deferred_32(float *c, const float *a, const float *b,
                          unsigned long int m, unsigned long int n,
                          unsigned long int p);

void fmatmul_4x4_fused_32(float *c, const float *a, const float *b,
                       const float *bias, unsigned long int m,
                       unsigned long int n, unsigned long int p);

void fmatmul_8x8_32(float *c, const float *a, const float *b,
                 unsigned long int m, unsigned long int n, unsigned long int p);
void fmatmul_vec_8x8_slice_init_32();
void fmatmul_vec_8x8_slice_load_bias_32(const float *bias_slice);
void fmatmul_vec_8x8_slice_init_fused_32(const float *bias_slice);
void fmatmul_vec_8x8_32(float *c, const float *a, const float *b,
                     unsigned long int n, unsigned long int p);

void fmatmul_8x8_nt_32(float *c, const float *a, const float *b,
                    unsigned long int m, unsigned long int n,
                    unsigned long int p);
void fmatmul_vec_8x8_nt_32(float *c, const float *a, const float *b,
                        unsigned long int n, unsigned long int p);
void fmatmul_8x8_tn_32(float *c, const float *a, const float *b,
                    unsigned long int m, unsigned long int n,
                    unsigned long int p);
void fmatmul_vec_8x8_tn_32(float *c, const float *a, const float *b,
                        unsigned long int n, unsigned long int p,
                        unsigned long int lda);

void fmatmul_8x8_fused_32(float *c, const float *a, const float *b,
                       const float *bias, unsigned long int m,
                       unsigned long int n, unsigned long int p);

void fmatmul_16x16_32(float *c, const float *a, const float *b,
                   unsigned long int m, unsigned long int n,
                   unsigned long int p);
void fmatmul_vec_16x16_slice_init_32();
void fmatmul_vec_16x16_slice_load_bias_32(const float *bias_slice);
void fmatmul_vec_16x16_slice_init_fused_32(const float *bias_slice);
void fmatmul_vec_16x16_32(float *c, const float *a, const float *b,
                       unsigned long int n, unsigned long int p);

void fmatmul_16x16_nt_32(float *c, const float *a, const float *b,
                      unsigned long int m, unsigned long int n,
                      unsigned long int p);
void fmatmul_vec_16x16_nt_32(float *c, const float *a, const float *b,
                          unsigned long int n, unsigned long int p);
void fmatmul_16x16_tn_32(float *c, const float *a, const float *b,
                      unsigned long int m, unsigned long int n,
                      unsigned long int p);
void fmatmul_vec_16x16_tn_32(float *c, const float *a, const float *b,
                          unsigned long int n, unsigned long int p,
                          unsigned long int lda);

void fmatmul_16x16_fused_32(float *c, const float *a, const float *b,
                         const float *bias, unsigned long int m,
                         unsigned long int n, unsigned long int p);

void calc_bias_gradient_vec_32(float *d_bias, const float *d_output, int out_units, int batchsize);


#define DELTA 0.000001

extern int64_t event_trigger;

#endif
