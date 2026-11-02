// Copyright 2020 ETH Zurich and University of Bologna.
//
// SPDX-License-Identifier: Apache-2.0
//
// Author: Matteo Perotti (original 7x7), adapted for 3x3 float

#include "fconv3d.h"
#include <stdint.h>

static void fconv3d_CHx3x3_block_f32(float *o, const float *i, const float *f,
                                     int64_t M, int64_t N, int64_t n_,
                                     int64_t C, float bias)
{
    int64_t ldo = N << 2;
    int64_t ldi_pad = (N + 2) << 2; // N + F - 1 with F=3
    int64_t ich_len = (M + 2) * (N + 2);
    int64_t fch_len = 9;

    const float *i_row = i;
    float *o_row = o;

    for (int64_t m = 0; m < M; ++m) { //m because of padding
        // v20 accumulates the fc[0,3,6] taps (seeded with bias); v22 and v24
        // accumulate the fc[1,4,7] and fc[2,5,8] taps. All three MUST be reset
        // every output row -- v22/v24 were previously left un-zeroed, so they
        // carried a running sum across rows (a ~28x blow-up + per-row ramp in
        // the conv output). bias belongs only in v20.
        asm volatile("vfmv.v.f v20, %0" :: "f"(bias));
        asm volatile("vmv.v.i v22, 0");
        asm volatile("vmv.v.i v24, 0");

        for (int64_t ch = 0; ch < C; ++ch) {
            const float *row0 = i_row + ch * ich_len;
            const float *row1 = row0 + (N + 2);
            const float *row2 = row1 + (N + 2);

            const float *slide0 = row0 + n_;
            const float *slide1 = row1 + n_;
            const float *slide2 = row2 + n_;

            asm volatile("vle32.v v0,  (%0)" :: "r"(row0));
            asm volatile("vle32.v v6,  (%0)" :: "r"(row1));
            asm volatile("vle32.v v12, (%0)" :: "r"(row2));

            asm volatile("vfslide1down.vf v2,  v0,  %0" :: "f"(*slide0++));
            asm volatile("vfslide1down.vf v4,  v2,  %0" :: "f"(*slide0++)); //2 slides for the weight sees all the right pixels
            asm volatile("vfslide1down.vf v8,  v6,  %0" :: "f"(*slide1++));
            asm volatile("vfslide1down.vf v10, v8,  %0" :: "f"(*slide1++));
            asm volatile("vfslide1down.vf v14, v12, %0" :: "f"(*slide2++));
            asm volatile("vfslide1down.vf v16, v14, %0" :: "f"(*slide2++));

            const float *fc = f + ch * fch_len;
            asm volatile("vfmacc.vf v20, %0, v0"  :: "f"(fc[0]));
            asm volatile("vfmacc.vf v22, %0, v2"  :: "f"(fc[1]));
            asm volatile("vfmacc.vf v24, %0, v4"  :: "f"(fc[2]));
            asm volatile("vfmacc.vf v20, %0, v6"  :: "f"(fc[3]));
            asm volatile("vfmacc.vf v22, %0, v8"  :: "f"(fc[4]));
            asm volatile("vfmacc.vf v24, %0, v10" :: "f"(fc[5]));
            asm volatile("vfmacc.vf v20, %0, v12" :: "f"(fc[6]));
            asm volatile("vfmacc.vf v22, %0, v14" :: "f"(fc[7]));
            asm volatile("vfmacc.vf v24, %0, v16" :: "f"(fc[8]));
        }

        // asm volatile("vfadd.vf v20, v20, %0" :: "f"(bias));
        asm volatile("vfadd.vv v20, v20, v22"); // v20 = v20 + v22
        asm volatile("vfadd.vv v20, v20, v24"); // v20 = v20 + v24
        asm volatile("vse32.v v20, (%0)" :: "r"(o_row));

        o_row = (float *)((uintptr_t)o_row + ldo);
        i_row = (const float *)((uintptr_t)i_row + ldi_pad);
    }
}

void fconv3d_CHx3x3_f32(float *o, const float *i, const float *f, int64_t M,
                        int64_t N, int64_t C, float bias)
{
    unsigned long int block_size_n;

    asm volatile("vsetvli %0, %1, e32, m2, ta, ma" : "=r"(block_size_n) : "r"(N));

    for (unsigned long int n = 0; n < (unsigned long int)N; n += block_size_n) {
        const unsigned long int n_ = MIN((unsigned long int)N - n, block_size_n);

        const float *i_ = i + n;
        float *o_ = o + n;

        asm volatile("vsetvli zero, %0, e32, m2, ta, ma" :: "r"(n_));

        fconv3d_CHx3x3_block_f32(o_, i_, f, M, N, (int64_t)n_, C, bias);
    }
}
