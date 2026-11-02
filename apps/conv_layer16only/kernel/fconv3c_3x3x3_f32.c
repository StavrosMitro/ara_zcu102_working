// Copyright 2020 ETH Zurich and University of Bologna.
//
// SPDX-License-Identifier: Apache-2.0
//
// Author: Matteo Perotti (original 7x7), adapted for 3x3 _Float16

#include "fconv3d.h"
#include <stdint.h>

static void fconv3d_CHx3x3_block_f16(_Float16 *o, const _Float16 *i, const _Float16 *f,
                                     int64_t M, int64_t N, int64_t n_,
                                     int64_t C, _Float16 bias)
{
    int64_t ldo = N << 1;           // output row stride: N × sizeof(_Float16)
    int64_t ldi_pad = (N + 2) << 1; // input  row stride: (N+2) × sizeof(_Float16)
    int64_t ich_len = (M + 2) * (N + 2);
    int64_t fch_len = 9;

    const _Float16 *i_row = i;
    _Float16 *o_row = o;

    for (int64_t m = 0; m < M; ++m) { //m because of padding
        // asm volatile("vmv.v.i v20, 0");
        asm volatile("vfmv.v.f v20, %0" :: "f"(bias));
        asm volatile("vmv.v.i v22, 0");
        asm volatile("vmv.v.i v24, 0");


        for (int64_t ch = 0; ch < C; ++ch) {
            const _Float16 *row0 = i_row + ch * ich_len;
            const _Float16 *row1 = row0 + (N + 2);
            const _Float16 *row2 = row1 + (N + 2);

            const _Float16 *slide0 = row0 + n_;
            const _Float16 *slide1 = row1 + n_;
            const _Float16 *slide2 = row2 + n_;

            const _Float16 *fc = f + ch * fch_len;

            asm volatile("vle16.v v12, (%0)" :: "r"(row2));

            asm volatile("vle16.v v0,  (%0)" :: "r"(row0));
            asm volatile("vfslide1down.vf v2,  v0,  %0" :: "f"(*slide0++));
            asm volatile("vfmacc.vf v20, %0, v0"  :: "f"(fc[0]));

            asm volatile("vfslide1down.vf v4,  v2,  %0" :: "f"(*slide0++)); //2 slides for the weight sees all the right pixels
            asm volatile("vfmacc.vf v22, %0, v2"  :: "f"(fc[1]));
            asm volatile("vfmacc.vf v24, %0, v4"  :: "f"(fc[2]));

            asm volatile("vle16.v v6,  (%0)" :: "r"(row1));
            asm volatile("vfslide1down.vf v8,  v6,  %0" :: "f"(*slide1++));
            asm volatile("vfmacc.vf v20, %0, v6"  :: "f"(fc[3]));
            asm volatile("vfmacc.vf v22, %0, v8"  :: "f"(fc[4]));

            asm volatile("vfslide1down.vf v10, v8,  %0" :: "f"(*slide1++));
            asm volatile("vfmacc.vf v24, %0, v10" :: "f"(fc[5]));

            asm volatile("vfslide1down.vf v14, v12, %0" :: "f"(*slide2++));
            asm volatile("vfmacc.vf v20, %0, v12" :: "f"(fc[6]));

            asm volatile("vfslide1down.vf v16, v14, %0" :: "f"(*slide2++));
            asm volatile("vfmacc.vf v22, %0, v14" :: "f"(fc[7]));
            asm volatile("vfmacc.vf v24, %0, v16" :: "f"(fc[8]));
            
        }
        asm volatile("vfadd.vv v20,v20,v22");
        asm volatile("vfadd.vv v20,v20,v24");

        // asm volatile("vfadd.vf v20, v20, %0" :: "f"(bias));
        asm volatile("vse16.v v20, (%0)" :: "r"(o_row));

        o_row = (_Float16 *)((uintptr_t)o_row + ldo);
        i_row = (const _Float16 *)((uintptr_t)i_row + ldi_pad);
    }
}

void fconv3d_CHx3x3_f16(_Float16 *o, const _Float16 *i, const _Float16 *f, int64_t M,
                        int64_t N, int64_t C, _Float16 bias)
{
    unsigned long int block_size_n;

    asm volatile("vsetvli %0, %1, e16, m2, ta, ma" : "=r"(block_size_n) : "r"(N));

    for (unsigned long int n = 0; n < (unsigned long int)N; n += block_size_n) {
        const unsigned long int n_ = MIN((unsigned long int)N - n, block_size_n);

        const _Float16 *i_ = i + n;
        _Float16 *o_ = o + n;

        asm volatile("vsetvli zero, %0, e16, m2, ta, ma" :: "r"(n_));

        fconv3d_CHx3x3_block_f16(o_, i_, f, M, N, (int64_t)n_, C, bias);
    }
}

// Alternative dataflow: instead of vsetvli'ing to N and processing one output
// row per outer iteration (fconv3d_CHx3x3_f16 above), treat the whole padded
// (M+2)xW input (W = N+2) as ONE flat stream and walk it in VLMAX-sized (m4)
// chunks that ignore row boundaries entirely. Each chunk computes taps for
// every lane -- including the 2 padding columns per row and any row-boundary
// crossing inside the chunk -- then vcompress.vm discards the invalid lanes
// and vcpop.m gives the exact store length.
//
// Why the horizontal taps stay correct across row boundaries: output column c
// only ever needs padded-input columns c, c+1, c+2, and since c <= N-1, c+2 <=
// N+1 = W-1 -- i.e. never past the END of that column's OWN row. So the two
// vfslide1down scalars fetched past a chunk's tail are always the correct
// next elements of the SAME logical row for any lane whose output is actually
// kept; lanes where that isn't true are exactly the ones the mask drops.
//
// Register plan (m4 = 4 regs/group, only 8 aligned group bases exist:
// v0/4/8/12/16/20/24/28) -- one group per row-temp/accumulator/mask/column
// slot, all 8 in permanent use, so row temps (v4/v8/v12) are computed and
// consumed one row at a time instead of the 9-group unrolled layout the m2
// block kernel above uses (that layout doesn't fit at m4).
void fconv3d_CHx3x3_f16_stream(_Float16 *o, const _Float16 *i, const _Float16 *f,
                               int64_t M, int64_t N, int64_t C, _Float16 bias)
{
    const int64_t W = N + 2;
    const int64_t total = M * W;
    const int64_t ich_len = (M + 2) * W;
    const int64_t fch_len = 9;

    unsigned long int vlmax;
    asm volatile("vsetvli %0, zero, e16, m4, ta, ma" : "=r"(vlmax));
    asm volatile("vid.v    v16");
    asm volatile("vremu.vx v16, v16, %0" :: "r"(W)); // v16 = column-within-row per lane

    const int64_t r = (int64_t)vlmax % W;

    _Float16 *o_ptr = o;
    unsigned long int vl;
    for (int64_t p = 0; p < total; p += (int64_t)vl) {
        const int64_t avl = MIN(total - p, (int64_t)vlmax);
        asm volatile("vsetvli %0, %1, e16, m4, ta, ma" : "=r"(vl) : "r"(avl));

        // Valid-output mask: column < N (drops the 2 right-padding columns/row).
        asm volatile("vmsltu.vx v0, v16, %0" :: "r"(N));

        asm volatile("vfmv.v.f v20, %0" :: "f"(bias));
        asm volatile("vmv.v.i  v24, 0");
        asm volatile("vmv.v.i  v28, 0");

        for (int64_t ch = 0; ch < C; ++ch) {
            const _Float16 *chan_base = i + ch * ich_len + p;
            const _Float16 *fc = f + ch * fch_len;

            const _Float16 *row0 = chan_base;
            const _Float16 *row1 = chan_base + W;
            const _Float16 *row2 = chan_base + 2 * W;

            const _Float16 *s0 = row0 + vl;
            const _Float16 *s1 = row1 + vl;
            const _Float16 *s2 = row2 + vl;

            asm volatile("vle16.v v4, (%0)" :: "r"(row0));
            asm volatile("vfslide1down.vf v8,  v4, %0" :: "f"(*s0++));
            asm volatile("vfslide1down.vf v12, v8, %0" :: "f"(*s0++));
            asm volatile("vfmacc.vf v20, %0, v4"  :: "f"(fc[0]));
            asm volatile("vfmacc.vf v24, %0, v8"  :: "f"(fc[1]));
            asm volatile("vfmacc.vf v28, %0, v12" :: "f"(fc[2]));

            asm volatile("vle16.v v4, (%0)" :: "r"(row1));
            asm volatile("vfslide1down.vf v8,  v4, %0" :: "f"(*s1++));
            asm volatile("vfslide1down.vf v12, v8, %0" :: "f"(*s1++));
            asm volatile("vfmacc.vf v20, %0, v4"  :: "f"(fc[3]));
            asm volatile("vfmacc.vf v24, %0, v8"  :: "f"(fc[4]));
            asm volatile("vfmacc.vf v28, %0, v12" :: "f"(fc[5]));

            asm volatile("vle16.v v4, (%0)" :: "r"(row2));
            asm volatile("vfslide1down.vf v8,  v4, %0" :: "f"(*s2++));
            asm volatile("vfslide1down.vf v12, v8, %0" :: "f"(*s2++));
            asm volatile("vfmacc.vf v20, %0, v4"  :: "f"(fc[6]));
            asm volatile("vfmacc.vf v24, %0, v8"  :: "f"(fc[7]));
            asm volatile("vfmacc.vf v28, %0, v12" :: "f"(fc[8]));
        }

        asm volatile("vfadd.vv v20, v20, v24");
        asm volatile("vfadd.vv v20, v20, v28");

        // Roll the column pattern forward by (vlmax mod W) without a divide;
        // v4/v8 are free again here, everything they held this iteration has
        // already been consumed above.
        asm volatile("vadd.vx  v4, v16, %0" :: "r"(r));
        asm volatile("vsub.vx  v8, v4, %0" :: "r"(W));
        asm volatile("vminu.vv v16, v4, v8");

        unsigned long int nv;
        asm volatile("vcompress.vm v4, v20, v0");
        asm volatile("vcpop.m %0, v0" : "=r"(nv));
        asm volatile("vsetvli zero, %0, e16, m4, ta, ma" :: "r"(nv));
        asm volatile("vse16.v v4, (%0)" :: "r"(o_ptr));
        o_ptr += nv;
    }
}
