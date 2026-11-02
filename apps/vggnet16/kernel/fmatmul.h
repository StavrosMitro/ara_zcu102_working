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

#ifndef FMATMUL_H
#define FMATMUL_H

#include <stdint.h>
#include <stddef.h>

// ---------------------------------------------------------------------------
// Hardware CSR macros — Pure FP16 training support
// ---------------------------------------------------------------------------
// Round-to-Nearest-Even: minimises bias accumulation in long FP16 dot-products.
#define SET_FRM_RNE()       asm volatile("csrw frm, zero")
// Clear all FP exception flags before a guarded backward-pass section.
#define CLEAR_FFLAGS()      asm volatile("csrw fflags, zero")
// Read FP exception flags into an unsigned int lvalue.
#define READ_FFLAGS(flags)  asm volatile("csrr %0, fflags" : "=r"(flags))
// Bit 2 of fflags = Overflow flag (OF).
#define FFLAG_OVERFLOW_MASK  0x04u
// Bit 1 of fflags = Underflow flag (UF). Set only when the result is both tiny
// AND inexact (gradual-underflow HW: an exactly-representable subnormal does not
// raise it), so it is a usable — if noisy — signal that gradients are decaying.
#define FFLAG_UNDERFLOW_MASK 0x02u

// ---------------------------------------------------------------------------
// FP16 helper: in-place vectorised scalar multiply (m8 tile loop, e16 SEW)
// ---------------------------------------------------------------------------
void vector_scale_fp16(_Float16 *arr, _Float16 scale, size_t n);

// ---------------------------------------------------------------------------
// FP16 matmul dispatch (selects tiling strategy by M)
// ---------------------------------------------------------------------------
void fmatmul_32(_Float16 *c, const _Float16 *a, const _Float16 *b,
                unsigned long int m, unsigned long int n, unsigned long int p);

void fmatmul_fused_32(_Float16 *c, const _Float16 *a, const _Float16 *b,
                      const _Float16 *bias, unsigned long int m,
                      unsigned long int n, unsigned long int p);

void fmatmul_nt_32(_Float16 *c, const _Float16 *a, const _Float16 *b,
                   unsigned long int m, unsigned long int n, unsigned long int p);

void fmatmul_tn_32(_Float16 *c, const _Float16 *a, const _Float16 *b,
                   unsigned long int m, unsigned long int n, unsigned long int p);

// ---------------------------------------------------------------------------
// 4x4 tile
// ---------------------------------------------------------------------------
void fmatmul_4x4_32(_Float16 *c, const _Float16 *a, const _Float16 *b,
                    unsigned long int m, unsigned long int n, unsigned long int p);
void fmatmul_vec_4x4_slice_init_32(void);
void fmatmul_vec_4x4_slice_load_bias_32(const _Float16 *bias_slice);
void fmatmul_vec_4x4_slice_init_fused_32(const _Float16 *bias_slice);
void fmatmul_vec_4x4_32(_Float16 *c, const _Float16 *a, const _Float16 *b,
                        unsigned long int n, unsigned long int p);
void fmatmul_4x4_nt_32(_Float16 *c, const _Float16 *a, const _Float16 *b,
                       unsigned long int m, unsigned long int n, unsigned long int p);
void fmatmul_vec_4x4_slice_init_nt_32(void);
void fmatmul_vec_4x4_nt_32(_Float16 *c, const _Float16 *a, const _Float16 *b,
                            unsigned long int n, unsigned long int p);
void fmatmul_4x4_tn_32(_Float16 *c, const _Float16 *a, const _Float16 *b,
                       unsigned long int m, unsigned long int n, unsigned long int p);
void fmatmul_vec_4x4_slice_init_tn_32(void);
void fmatmul_vec_4x4_tn_32(_Float16 *c, const _Float16 *a, const _Float16 *b,
                            unsigned long int m, unsigned long int p,
                            unsigned long int n);
void fmatmul_4x4_deferred_32(_Float16 *c, const _Float16 *a, const _Float16 *b,
                              unsigned long int m, unsigned long int n,
                              unsigned long int p);
void fmatmul_vec_4x4_slice_init_deferred_32(void);
void fmatmul_vec_4x4_deferred_32(_Float16 *c, const _Float16 *a, const _Float16 *b,
                                  unsigned long int n, unsigned long int p,
                                  unsigned long int p_);
void fmatmul_4x4_fused_32(_Float16 *c, const _Float16 *a, const _Float16 *b,
                           const _Float16 *bias, unsigned long int m,
                           unsigned long int n, unsigned long int p);

// ---------------------------------------------------------------------------
// 8x8 tile
// ---------------------------------------------------------------------------
void fmatmul_8x8_32(_Float16 *c, const _Float16 *a, const _Float16 *b,
                    unsigned long int m, unsigned long int n, unsigned long int p);
void fmatmul_vec_8x8_slice_init_32(void);
void fmatmul_vec_8x8_slice_load_bias_32(const _Float16 *bias_slice);
void fmatmul_vec_8x8_slice_init_fused_32(const _Float16 *bias_slice);
void fmatmul_vec_8x8_32(_Float16 *c, const _Float16 *a, const _Float16 *b,
                        unsigned long int n, unsigned long int p);
void fmatmul_8x8_nt_32(_Float16 *c, const _Float16 *a, const _Float16 *b,
                       unsigned long int m, unsigned long int n, unsigned long int p);
void fmatmul_vec_8x8_slice_init_nt_32(void);
void fmatmul_vec_8x8_nt_32(_Float16 *c, const _Float16 *a, const _Float16 *b,
                            unsigned long int n, unsigned long int p);
void fmatmul_8x8_tn_32(_Float16 *c, const _Float16 *a, const _Float16 *b,
                       unsigned long int m, unsigned long int n, unsigned long int p);
void fmatmul_vec_8x8_slice_init_tn_32(void);
void fmatmul_vec_8x8_tn_32(_Float16 *c, const _Float16 *a, const _Float16 *b,
                            unsigned long int m, unsigned long int p,
                            unsigned long int n);
void fmatmul_8x8_fused_32(_Float16 *c, const _Float16 *a, const _Float16 *b,
                           const _Float16 *bias, unsigned long int m,
                           unsigned long int n, unsigned long int p);

// ---------------------------------------------------------------------------
// 16x16 tile
// ---------------------------------------------------------------------------
void fmatmul_16x16_32(_Float16 *c, const _Float16 *a, const _Float16 *b,
                      unsigned long int m, unsigned long int n, unsigned long int p);
void fmatmul_vec_16x16_slice_init_32(void);
void fmatmul_vec_16x16_slice_load_bias_32(const _Float16 *bias_slice);
void fmatmul_vec_16x16_slice_init_fused_32(const _Float16 *bias_slice);
void fmatmul_vec_16x16_32(_Float16 *c, const _Float16 *a, const _Float16 *b,
                          unsigned long int n, unsigned long int p);
void fmatmul_16x16_nt_32(_Float16 *c, const _Float16 *a, const _Float16 *b,
                         unsigned long int m, unsigned long int n, unsigned long int p);
void fmatmul_vec_16x16_slice_init_nt_32(void);
void fmatmul_vec_16x16_nt_32(_Float16 *c, const _Float16 *a, const _Float16 *b,
                              unsigned long int n, unsigned long int p);
void fmatmul_16x16_tn_32(_Float16 *c, const _Float16 *a, const _Float16 *b,
                         unsigned long int m, unsigned long int n, unsigned long int p);
void fmatmul_vec_16x16_slice_init_tn_32(void);
void fmatmul_vec_16x16_tn_32(_Float16 *c, const _Float16 *a, const _Float16 *b,
                              unsigned long int m, unsigned long int p,
                              unsigned long int n);
void fmatmul_16x16_fused_32(_Float16 *c, const _Float16 *a, const _Float16 *b,
                             const _Float16 *bias, unsigned long int m,
                             unsigned long int n, unsigned long int p);

// ---------------------------------------------------------------------------
// Bias gradient reduction
// ---------------------------------------------------------------------------
void calc_bias_gradient_vec_32(_Float16 *d_bias, const _Float16 *d_output,
                                int out_units, int batchsize);

// ---------------------------------------------------------------------------
// Scratch buffers (defined in fmatmul.c)
// ---------------------------------------------------------------------------
extern _Float16 fmatmul_c_scratch_32[];
extern _Float16 shared_memory_pool_32[];

#define DELTA 0.000001

extern int64_t event_trigger;

#endif // FMATMUL_H
