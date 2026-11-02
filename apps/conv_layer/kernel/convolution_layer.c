//
// File:        convolution_layer.c
// Description: Mixed-precision (FP16 compute / FP32 accumulate) convolution layer
//
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stddef.h>
#ifdef SPIKE
#include "printf.h"
#elif defined ARA_LINUX
#include <stdio.h>
#else
#include "printf.h"
#endif
#include "convolution_layer.h"
#include "matrix.h"
#include "runtime.h"
#include "fconv3d.h"
#include "fmatmul.h"
#ifndef MIN
#define MIN(a,b) (((a) < (b)) ? (a) : (b))
#endif

// =========================================================================
// Gather-based vectorized img2col support
// =========================================================================
#define MAX_GATHER_ELEMENTS (CONV_MAX_IKK * CONV_MAX_OWOH)

static uint32_t gather_offsets_buf[MAX_GATHER_ELEMENTS];      // FP32 byte offsets
static uint32_t gather_offsets_f16_buf[MAX_GATHER_ELEMENTS];  // FP16 byte offsets (= FP32 >> 1)
static int total_gather_elements = 0;

static inline void memset_vectorized_zero_f32(float *dst, size_t n_elements)
{
    asm volatile("vsetvli zero, zero, e32, m8, ta, ma");
    asm volatile("vmv.v.i v16, 0");
    while (n_elements > 0) {
        size_t vl;
        asm volatile("vsetvli %0, %1, e32, m8, ta, ma" : "=r"(vl) : "r"(n_elements));
        asm volatile("vse32.v v16, (%0)" :: "r"(dst));
        dst += vl;
        n_elements -= vl;
    }
}

static inline void memset_vectorized_zero_f16(_Float16 *dst, size_t n_elements)
{
    asm volatile("vsetvli zero, zero, e16, m8, ta, ma");
    asm volatile("vmv.v.i v16, 0");
    while (n_elements > 0) {
        size_t vl;
        asm volatile("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(n_elements));
        asm volatile("vse16.v v16, (%0)" :: "r"(dst));
        dst += vl;
        n_elements -= vl;
    }
}

// FP32 -> FP16 narrowing via vfncvt (e32,m8 load -> e16,m4 store)
void convert_f32_to_f16_vec(const float *src, _Float16 *dst, size_t n)
{
    while (n > 0) {
        size_t vl;
        asm volatile("vsetvli %0, %1, e32, m8, ta, ma" : "=r"(vl) : "r"(n));
        asm volatile("vle32.v v8, (%0)" :: "r"(src));
        asm volatile("vsetvli zero, zero, e16, m4, ta, ma");
        asm volatile("vfncvt.f.f.w v0, v8");
        asm volatile("vse16.v v0, (%0)" :: "r"(dst));
        src += vl; dst += vl; n -= vl;
    }
}

// =========================================================================
// Gather-based vectorized img2col (FP32 — used by backward d_weights)
// =========================================================================

static void img2col_vectorized_f16(const _Float16 *img, _Float16 *col)
{
    int n = total_gather_elements;
    const uint32_t *idx_ptr = gather_offsets_f16_buf;
    const _Float16 *src_img = img;
    _Float16 *dst_col = col;

    while (n > 0) {
        size_t vl;
        // VLMAX(e32,m8) == VLMAX(e16,m4): load indices with e32,m8, then gather with e16,m4
        asm volatile("vsetvli %0, %1, e32, m8, ta, ma" : "=r"(vl) : "r"(n));
        asm volatile("vle32.v v0, (%0)" :: "r"(idx_ptr));
        asm volatile("vsetvli zero, zero, e16, m4, ta, ma");
        asm volatile("vluxei32.v v16, (%0), v0" :: "r"(src_img));
        asm volatile("vse16.v v16, (%0)" :: "r"(dst_col));
        dst_col += vl;
        idx_ptr += vl;
        n -= (int)vl;
    }
}

static void img2col_vectorized(const float *img, float *col)
{
    int n = total_gather_elements;
    const uint32_t *idx_ptr = gather_offsets_buf;
    const float *src_img = img;
    float *dst_col = col;

    while (n > 0) {
        size_t vl;
        asm volatile("vsetvli %0, %1, e32, m8, ta, ma" : "=r"(vl) : "r"(n));
        asm volatile("vle32.v v8, (%0)" :: "r"(idx_ptr));
        asm volatile("vluxei32.v v16, (%0), v8" :: "r"(src_img));
        asm volatile("vse32.v v16, (%0)" :: "r"(dst_col));
        dst_col += vl;
        idx_ptr += vl;
        n -= vl;
    }
}

// =========================================================================
// Precompute img2col offsets — called once before training
// =========================================================================
// Returns the cycles spent in the actual offset generation only. The DEBUG
// printf_ calls above the timer are excluded: in Spike each character is a
// blocking HTIF syscall (~5000 cyc/char), which otherwise dwarfs the real work.
int64_t precompute_img2col_offsets_static(const conv_op *op)
{
    int iwih = op->in_w * op->in_h;
    int kk   = op->kernel_size * op->kernel_size;
    int ikk  = op->in_channels * kk;

    total_gather_elements = op->out_h * op->out_w * ikk;

    printf_("DEBUG: precompute_img2col_offsets_static called\n");
    printf_("  out_h=%d, out_w=%d, in_channels=%d, kk=%d, ikk=%d\n",
            op->out_h, op->out_w, op->in_channels, kk, ikk);
    printf_("  total_gather_elements=%d, MAX_GATHER_ELEMENTS=%d\n",
            total_gather_elements, MAX_GATHER_ELEMENTS);

    if (total_gather_elements > MAX_GATHER_ELEMENTS) {
        printf_("FATAL ERROR: gather offsets need %d elements, max is %d\n",
                total_gather_elements, MAX_GATHER_ELEMENTS);
        exit(1);
    }

    int64_t _cyc0 = get_cycle_count();

    uint32_t patch_pattern[9];
    int p_idx = 0;
    for (int j = 0; j < op->kernel_size; j++)
        for (int i = 0; i < op->kernel_size; i++)
            patch_pattern[p_idx++] = (uint32_t)((i + j * op->in_w) * (int)sizeof(float));

    size_t vl;
    asm volatile("vsetvli %0, %1, e32, m1, ta, ma" : "=r"(vl) : "r"(kk));
    asm volatile("vle32.v v8, (%0)" :: "r"(patch_pattern));

    uint32_t *out_ptr = gather_offsets_buf;
    for (int st_y = 0; st_y < op->out_h * op->stride; st_y += op->stride) {
        for (int st_x = 0; st_x < op->out_w * op->stride; st_x += op->stride) {
            for (int in_c = 0; in_c < op->in_channels; in_c++) {
                uint32_t base = (uint32_t)((st_x + st_y * op->in_w + in_c * iwih) * (int)sizeof(float));
                asm volatile("vadd.vx v16, v8, %0" :: "r"(base));
                asm volatile("vse32.v v16, (%0)" :: "r"(out_ptr));
                out_ptr += kk;
            }
        }
    }

    // Derive FP16 byte offsets: FP16 elements are 2 bytes so offset = FP32 offset >> 1
    {
        int n = total_gather_elements;
        const uint32_t *src = gather_offsets_buf;
        uint32_t *dst = gather_offsets_f16_buf;
        while (n > 0) {
            size_t vl;
            asm volatile("vsetvli %0, %1, e32, m8, ta, ma" : "=r"(vl) : "r"(n));
            asm volatile("vle32.v v8, (%0)" :: "r"(src));
            asm volatile("vsrl.vi v8, v8, 1");
            asm volatile("vse32.v v8, (%0)" :: "r"(dst));
            src += vl; dst += vl; n -= (int)vl;
        }
    }

    return get_cycle_count() - _cyc0;
}

// =========================================================================
// Static scratch buffers
// =========================================================================
#define CONV_MAX_OUT_W 32
#define CONV_MAX_OUT_H 32
#define CONV_MAX_PADDED_OWOH ((CONV_MAX_OUT_W + 2) * (CONV_MAX_OUT_H + 2))

static float conv1_xcol_scratch[CONV1_XCOL_ELEMS];
static float conv2_xcol_scratch[16 * 16 * (64 * 3 * 3)];
static float conv3_xcol_scratch[8  * 8  * (128 * 3 * 3)];
static float conv4_xcol_scratch[8  * 8  * (256 * 3 * 3)];
static float conv5_input_col_full[CONV5_INPUT_COL_SIZE];

static float conv_t_dweights_scratch[CONV_MAX_T_DWEIGHTS];
static float conv_d_out_copy_scratch[CONV_MAX_DOCOPY];
static float conv_d_x_col_scratch[CONV_MAX_DXCOL];
static float conv_weights_t_scratch[CONV_MAX_T_DWEIGHTS];
static float conv_d_input_tmp_scratch[CONV_MAX_OC * CONV_MAX_OWOH];

// FP16 working buffers
static _Float16 conv3x3_filter_f16_scratch[CONV_MAX_IKK];
static _Float16 conv_d_out_padded_f16_scratch[CONV_MAX_OC * CONV_MAX_PADDED_OWOH];
static _Float16 x_col_f16_scratch[CONV1_XCOL_ELEMS];

// =========================================================================
// Forward declarations
// =========================================================================
static float *conv_forward_xcol_ptr(conv_op *op, short batch_id);
static void img2col(const float *img, float *col, const conv_op *op);
static void conv_op_forward_3x3_ara(conv_op *op);
static void pad_channels(float *dst, const float *src, int channels, int in_h, int in_w, int pad);
static void pad_channels_f16(_Float16 *dst, const _Float16 *src, int channels, int in_h, int in_w, int pad);
static void pack_conv3x3_filter_rot180_dx_f16(const conv_op *op, int ic, _Float16 *dst);
static void conv_op_backward_input_3x3_ara(conv_op *op);

// =========================================================================
// Verify scalar vs vectorized img2col
// =========================================================================
void verify_img2col_implementations(const conv_op *op, const float *test_input)
{
    float *col_scalar     = conv1_xcol_scratch;
    float *col_vectorized = conv2_xcol_scratch;

    printf_("\n========== img2col Implementation Verification ==========\n");
    img2col(test_input, col_scalar, op);
    img2col_vectorized(test_input, col_vectorized);

    int total_elements = op->out_h * op->out_w * op->in_channels * op->kernel_size * op->kernel_size;
    int diffs = 0;
    float max_diff = 0.0f;
    float sum_diff = 0.0f;

    for (int i = 0; i < total_elements; i++) {
        float diff = fabsf(col_scalar[i] - col_vectorized[i]);
        if (diff > 1e-6f) {
            diffs++;
            sum_diff += diff;
            if (diff > max_diff) max_diff = diff;
            if (diffs <= 10)
                printf_("  Diff at [%d]: scalar=%.6f, vect=%.6f, diff=%.6e\n",
                        i, col_scalar[i], col_vectorized[i], diff);
        }
    }
    if (diffs == 0)
        printf_("VERIFICATION PASSED: %d elements identical\n", total_elements);
    else
        printf_("VERIFICATION FAILED: %d/%d diffs, max=%.6e\n", diffs, total_elements, max_diff);
    printf_("==========================================================\n\n");
    (void)sum_diff;
}

// =========================================================================
// Filter packing helpers
// =========================================================================

// Weights layout: [ic*kk + ky*k + kx][oc] — for a fixed oc, elements sit
// at stride out_channels. vlse32 gathers them in one strided load.
static void pack_conv3x3_filter(const conv_op *op, int oc, float *dst)
{
    int n = op->in_channels * op->kernel_size * op->kernel_size;
    const float *src = op->weights + oc;
    int stride_bytes = op->out_channels * (int)sizeof(float);

    while (n > 0) {
        size_t vl;
        asm volatile("vsetvli %0, %1, e32, m8, ta, ma" : "=r"(vl) : "r"(n));
        asm volatile("vlse32.v v8, (%0), %1" :: "r"(src), "r"(stride_bytes));
        asm volatile("vse32.v v8, (%0)" :: "r"(dst));
        src += (int)vl * op->out_channels;
        dst += vl;
        n   -= (int)vl;
    }
}

// Same as above, FP16 source and destination.
static void pack_conv3x3_filter_f16(const conv_op *op, int oc, _Float16 *dst)
{
    int n = op->in_channels * op->kernel_size * op->kernel_size;
    const _Float16 *src = op->weights_f16 + oc;
    int stride_bytes = op->out_channels * (int)sizeof(_Float16);

    while (n > 0) {
        size_t vl;
        asm volatile("vsetvli %0, %1, e16, m4, ta, ma" : "=r"(vl) : "r"(n));
        asm volatile("vlse16.v v0, (%0), %1" :: "r"(src), "r"(stride_bytes));
        asm volatile("vse16.v v0, (%0)" :: "r"(dst));
        src += (int)vl * op->out_channels;
        dst += vl;
        n   -= (int)vl;
    }
}

// Rot-180 packing for backward d_input.
// Restructure loop to [ky][kx][oc]: for each kernel position, the OC source
// elements are contiguous (vle16), and the OC destination slots sit at
// stride kk*2 bytes (vsse16).
// Vectorized along the KERNEL axis (kk elements), not out_channels.
//
// dst[oc*kk + j] = weights_f16[(ic*kk + flip(j))*OC + oc], and for a square
// kernel flip(j) = kk-1-j, so for a fixed oc the source walks BACKWARDS with
// byte stride -OC*2 -- vlse16 handles that directly (RVV strides are signed).
//
// The previous form vectorized over out_channels, which runs at vl=1 when OC==1
// (the common case here) and is slower than a scalar loop.
static void pack_conv3x3_filter_rot180_dx_f16(const conv_op *op, int ic, _Float16 *dst)
{
    int k  = op->kernel_size;
    int kk = k * k;
    int OC = op->out_channels;
    int stride_bytes = -OC * (int)sizeof(_Float16);   // negative: kernel backwards

    for (int oc = 0; oc < OC; oc++) {
        const _Float16 *src = op->weights_f16 + (ic * kk + kk - 1) * OC + oc;
        _Float16 *d = dst + oc * kk;
        int n = kk;
        while (n > 0) {
            size_t vl;
            asm volatile("vsetvli %0, %1, e16, m4, ta, ma" : "=r"(vl) : "r"(n));
            asm volatile("vlse16.v v0, (%0), %1" :: "r"(src), "r"(stride_bytes));
            asm volatile("vse16.v v0, (%0)" :: "r"(d));
            src -= (int)vl * OC;
            d   += vl;
            n   -= (int)vl;
        }
    }
}

// =========================================================================
// Padding helpers
// =========================================================================
static void pad_channels(float *dst, const float *src, int channels, int in_h, int in_w, int pad)
{
    int out_h = in_h + 2 * pad;
    int out_w = in_w + 2 * pad;
    size_t total = (size_t)channels * (size_t)out_h * (size_t)out_w;
    memset_vectorized_zero_f32(dst, total);

    int in_plane  = in_h * in_w;
    int out_plane = out_h * out_w;
    for (int c = 0; c < channels; c++) {
        const float *src_c = src + c * in_plane;
        float *dst_c = dst + c * out_plane + pad * out_w + pad;
        for (int y = 0; y < in_h; y++) {
            const float *s = src_c + y * in_w;
            float *d = dst_c + y * out_w;
            int n = in_w;
            while (n > 0) { //memcpy
                size_t vl;
                asm volatile("vsetvli %0, %1, e32, m8, ta, ma" : "=r"(vl) : "r"(n));
                asm volatile("vle32.v v8, (%0)" :: "r"(s));
                asm volatile("vse32.v v8, (%0)" :: "r"(d));
                s += vl; d += vl; n -= (int)vl;
            }
        }
    }
}

// Convert-then-pad: src is unpadded FP16, dst is padded FP16
static void pad_channels_f16(_Float16 *dst, const _Float16 *src, int channels, int in_h, int in_w, int pad)
{
    int out_h = in_h + 2 * pad;
    int out_w = in_w + 2 * pad;
    size_t total = (size_t)channels * (size_t)out_h * (size_t)out_w;
    memset_vectorized_zero_f16(dst, total);

    int in_plane  = in_h * in_w;
    int out_plane = out_h * out_w;
    for (int c = 0; c < channels; c++) {
        const _Float16 *src_c = src + c * in_plane;
        _Float16 *dst_c = dst + c * out_plane + pad * out_w + pad;
        for (int y = 0; y < in_h; y++) {
            const _Float16 *s_ptr = src_c + y * in_w;
            _Float16 *d_ptr = dst_c + y * out_w;
            int n_copy = in_w;
            while (n_copy > 0) {
                size_t vl;
                asm volatile("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(n_copy));
                asm volatile("vle16.v v16, (%0)" :: "r"(s_ptr));
                asm volatile("vse16.v v16, (%0)" :: "r"(d_ptr));
                s_ptr += vl; d_ptr += vl; n_copy -= vl;
            }
        }
    }
}

// =========================================================================
// Capability checks
// =========================================================================
static int conv_can_use_3x3_dx(const conv_op *op)
{
    if (op->kernel_size != 3 || op->stride != 1) return 0;
    if (op->in_w != op->out_w + 2 || op->in_h != op->out_h + 2) return 0;
    if ((size_t)op->out_channels * 9 > (size_t)CONV_MAX_IKK) return 0;
    return 1;
}

// =========================================================================
// Mixed-precision forward: FP16 input + FP16 weights -> FP32 output
// =========================================================================
static void conv_op_forward_3x3_ara(conv_op *op)
{
    int out_plane    = op->out_w * op->out_h;
    int in_channels  = op->in_channels;
    int out_channels = op->out_channels;

    for (int b = 0; b < op->batchsize; b++) {
        const _Float16 *input_f16_b = op->input_f16 + b * op->in_units;
        _Float16 *output_f16_b = op->output_f16 + b * op->out_units;

        for (int oc = 0; oc < out_channels; oc++) {
            pack_conv3x3_filter_f16(op, oc, conv3x3_filter_f16_scratch);
            _Float16 *out_oc = output_f16_b + oc * out_plane;
            fconv3d_CHx3x3_f16_f16out(out_oc, input_f16_b, conv3x3_filter_f16_scratch,
                                      op->out_h, op->out_w, in_channels, op->bias[oc]);
        }
    }
}

// =========================================================================
// Mixed-precision backward d_input: convert-then-pad for d_output
// =========================================================================
static void conv_op_backward_input_3x3_ara(conv_op *op)
{
    int pad        = (op->in_w - op->out_w) / 2;
    int out_plane  = op->out_w * op->out_h;
    int padded_plane = op->in_w * op->in_h;

    if (pad <= 0) { printf_("Error: invalid padding for 3x3 d_input\n"); exit(1); }
    if ((size_t)op->out_channels * (size_t)padded_plane > (size_t)CONV_MAX_OC * (size_t)CONV_MAX_PADDED_OWOH) {
        printf_("Error: d_output padded workspace overflow\n"); exit(1);
    }
    if ((size_t)op->in_channels * (size_t)out_plane > (size_t)CONV_MAX_OC * (size_t)CONV_MAX_OWOH) {
        printf_("Error: d_input workspace overflow\n"); exit(1);
    }

    for (int b = 0; b < op->batchsize; b++) {
        // d_output already in FP16 — pad directly
        pad_channels_f16(conv_d_out_padded_f16_scratch, op->d_output_f16 + b * op->out_units,
                         op->out_channels, op->out_h, op->out_w, pad);

        float *d_input_tmp = conv_d_input_tmp_scratch;
        for (int ic = 0; ic < op->in_channels; ic++) {
            pack_conv3x3_filter_rot180_dx_f16(op, ic, conv3x3_filter_f16_scratch);
            float *d_in_ch = d_input_tmp + ic * out_plane;
            fconv3d_CHx3x3_f16(d_in_ch, conv_d_out_padded_f16_scratch,
                               conv3x3_filter_f16_scratch,
                               op->out_h, op->out_w, op->out_channels, 0.0f);
        }

        // Write d_input into padded d_input buffer
        float *d_in_b = op->d_input + b * op->in_units;
        pad_channels(d_in_b, d_input_tmp, op->in_channels, op->out_h, op->out_w, pad);
    }
}

// =========================================================================
// xcol pointer helper
// =========================================================================
static float *conv_forward_xcol_ptr(conv_op *op, short batch_id)
{
    int col_size = (op->in_channels * op->kernel_size * op->kernel_size) * (op->out_w * op->out_h);
    if (op->layer_id == 5)
        return op->input_col + batch_id * col_size;
    switch (op->layer_id) {
        case 1: return conv1_xcol_scratch;
        case 2: return conv2_xcol_scratch;
        case 3: return conv3_xcol_scratch;
        case 4: return conv4_xcol_scratch;
        default:
            printf_("Error: invalid conv layer_id=%d\n", op->layer_id);
            exit(1);
    }
}

// =========================================================================
// Scalar img2col (FP32) — fallback and comparison
// =========================================================================
static void img2col(const float *img, float *col, const conv_op *op)
{
    int iwih = op->in_w * op->in_h;
    int kk   = op->kernel_size * op->kernel_size;
    int ikk  = op->in_channels * kk;

    for (int in_c = 0; in_c < op->in_channels; in_c++) {
        int out_y = 0;
        for (int st_y = 0; st_y < op->out_h * op->stride; st_y += op->stride, out_y++) {
            int out_x = 0;
            for (int st_x = 0; st_x < op->out_w * op->stride; st_x += op->stride, out_x++) {
                int patch_idx   = out_y * op->out_w + out_x;
                int x_col_offset = patch_idx * ikk + in_c * kk;
                for (int j = 0; j < op->kernel_size; j++)
                    for (int i = 0; i < op->kernel_size; i++) {
                        int input_offset = (st_x + i) + (st_y + j) * op->in_w + in_c * iwih;
                        col[x_col_offset++] = img[input_offset];
                    }
            }
        }
    }
}

// FP32 scalar forward (used by conv_op_forward_im2col for comparison)
static void conv_op_forward_single(conv_op *op, short batch_id)
{
    float *x_col    = conv_forward_xcol_ptr(op, batch_id);
    float *t_input  = op->input  + batch_id * op->in_units;
    float *t_output = op->output + batch_id * op->out_units;
    int ikk  = op->in_channels * op->kernel_size * op->kernel_size;
    int owoh = op->out_w * op->out_h;

    img2col(t_input, x_col, op);
    matrix_multiply(x_col, op->weights, t_output, owoh, ikk, op->out_channels);
    matrix_transpose(t_output, owoh, op->out_channels);

    int o_offset = 0;
    for (int i = 0; i < op->out_channels; i++) {
        float tmp = op->bias[i];
        while (o_offset < (i + 1) * owoh)
            t_output[o_offset++] += tmp;
    }
}

static void col2img(const float *col, float *img, const conv_op *op)
{
    int iwih = op->in_w * op->in_h;
    int kk   = op->kernel_size * op->kernel_size;
    int ikk  = op->in_channels * kk;

    int out_y = 0;
    for (int st_y = 0; st_y < op->out_h * op->stride; st_y += op->stride, out_y++) {
        int out_x = 0;
        for (int st_x = 0; st_x < op->out_w * op->stride; st_x += op->stride, out_x++) {
            for (int in_c = 0; in_c < op->in_channels; in_c++) {
                int patch_idx    = out_y * op->out_w + out_x;
                int x_col_offset = patch_idx * ikk + in_c * kk;
                for (int j = 0; j < op->kernel_size; j++)
                    for (int i = 0; i < op->kernel_size; i++) {
                        int input_offset = (st_x + i) + (st_y + j) * op->in_w + in_c * iwih;
                        img[input_offset] += col[x_col_offset++];
                    }
            }
        }
    }
}

// =========================================================================
// Public forward interface
// =========================================================================
void conv_op_forward(conv_op *op)
{
    if (op->layer_id == 5) {
        if (op->batchsize > ALEXNET_STATIC_MAX_BATCH) {
            printf_("Error: conv5 batchsize %d exceeds static max %d\n",
                    op->batchsize, ALEXNET_STATIC_MAX_BATCH);
            exit(1);
        }
        op->input_col = conv5_input_col_full;
        // NOTE: element count, NOT bytes -- do not re-add * sizeof(float).
        // The libc memset in common/string.c drops to a BYTE loop unless dest is
        // 8-byte aligned. This buffer happens to be 8-byte aligned in this build,
        // but conv_layer32's identical buffer is not, where the same memset cost
        // 9249 cycles. vse32 needs only 4-byte alignment, so this has no cliff.
        memset_vectorized_zero_f32(op->input_col,
               (size_t)op->batchsize *
               (size_t)(op->in_channels * op->kernel_size * op->kernel_size) *
               (size_t)(op->out_w * op->out_h));
    } else {
        op->input_col = NULL;
    }
    conv_op_forward_3x3_ara(op);
}

void conv_op_forward_im2col(conv_op *op)
{
    if (op->layer_id == 5 && op->input_col == NULL) {
        printf_("Error: conv_op_forward_im2col requires input_col for layer_id=5\n");
        exit(1);
    }
    for (int p = 0; p < op->batchsize; p++)
        conv_op_forward_single(op, (short)p);
}

// =========================================================================
// Mixed-precision backward: d_weights uses fmatmul (FP16 x FP16 -> FP32)
// =========================================================================
void conv_op_backward_full_profile(conv_op *op, conv_backward_cycle_breakdown *cycles)
{
    int oc   = op->out_channels;
    int ikk  = op->in_channels * op->kernel_size * op->kernel_size;
    int owoh = op->out_w * op->out_h;
    int64_t t0 = 0;

    if (cycles) {
        cycles->d_input_cycles          = 0;
        cycles->d_bias_cycles           = 0;
        cycles->d_weights_im2col_cycles = 0;
        cycles->d_weights_cycles        = 0;
    }

    float *t_d_weights = conv_t_dweights_scratch;
    int64_t dweights_total_t0 = get_cycle_count();

    for (int p = 0; p < op->batchsize; p++) {
        t0 = get_cycle_count();
        // DISABLED (hang suspect + dead work). With the d_weights GEMM below
        // commented out, nothing reads x_col_f16_scratch. It is also what HUNG
        // the FP16 app on Ara RTL: vluxei32.v indexed gather at e16/m4 (index
        // EEW=32 vs data EEW=16 => index EMUL=8). Spike never showed it.
        // Re-enable together with the GEMM.
        // img2col_vectorized_f16(op->input_f16 + p * op->in_units, x_col_f16_scratch);
        if (cycles) cycles->d_weights_im2col_cycles += get_cycle_count() - t0;

        memset_vectorized_zero_f32(t_d_weights, (size_t)oc * (size_t)ikk);

        t0 = get_cycle_count();

        // DISABLED: benchmarking the conv kernel, not the weight-gradient GEMM.
        // t_d_weights stays zeroed by the memset above, so the scatter below
        // accumulates zeros and d_weights stays 0. Uncomment to restore.
        //
        // Widening FMA matmul: FP16 x FP16 -> FP32 accumulate.
        // Goes through matrix_multiply_f16 rather than fmatmul directly: with
        // oc < the microkernel row block (e.g. oc=1 -> fmatmul_4x4), a raw
        // fmatmul reads/writes past the M=oc buffers. The wrapper row-pads M.
        // matrix_multiply_f16(op->d_output_f16 + p * oc * owoh, x_col_f16_scratch,
        //                     t_d_weights, oc, owoh, ikk);

        // Scatter-accumulate t_d_weights[oc,ikk] -> op->d_weights[ikk,oc]
        int stride_bytes = oc * (int)sizeof(float);
        for (int j = 0; j < oc; j++) {
            float *src = t_d_weights + j * ikk;
            float *dst = op->d_weights + j;
            int n = ikk;
            while (n > 0) {
                size_t vl;
                asm volatile("vsetvli %0, %1, e32, m8, ta, ma" : "=r"(vl) : "r"(n));
                asm volatile("vle32.v v8,  (%0)" :: "r"(src));
                asm volatile("vlse32.v v16, (%0), %1" :: "r"(dst), "r"(stride_bytes));
                asm volatile("vfadd.vv v16, v16, v8");
                asm volatile("vsse32.v v16, (%0), %1" :: "r"(dst), "r"(stride_bytes));
                src += vl;
                dst += (int)vl * oc;
                n   -= (int)vl;
            }
        }
    }
    if (cycles) cycles->d_weights_cycles += get_cycle_count() - dweights_total_t0;

    // Average d_weights over batch
    float inv_batch = 1.0f / (float)op->batchsize;
    int dw_elems = oc * ikk;
    float *dw_ptr = op->d_weights;
    while (dw_elems > 0) {
        size_t vl;
        asm volatile("vsetvli %0, %1, e32, m8, ta, ma" : "=r"(vl) : "r"(dw_elems));
        asm volatile("vle32.v v8, (%0)" :: "r"(dw_ptr));
        asm volatile("vfmul.vf v8, v8, %0" :: "f"(inv_batch));
        asm volatile("vse32.v v8, (%0)" :: "r"(dw_ptr));
        dw_ptr += vl;
        dw_elems -= (int)vl;
    }

    // d_bias: sum over batch and spatial, then average
    t0 = get_cycle_count();
    float inv_batchsize = 1.0f / (float)op->batchsize;
    for (int i = 0; i < oc; i++) {
        asm volatile("vsetvli zero, zero, e32, m8, ta, ma");
        asm volatile("vmv.v.i v8, 0");
        for (int p = 0; p < op->batchsize; p++) {
            const _Float16 *ptr = op->d_output_f16 + p * oc * owoh + i * owoh;
            int n = owoh;
            while (n > 0) {
                size_t vl;
                // Load FP16, widen to FP32, accumulate
                asm volatile("vsetvli %0, %1, e16, m4, ta, ma" : "=r"(vl) : "r"(n));
                asm volatile("vle16.v v0, (%0)" :: "r"(ptr));
                
                asm volatile("vfwcvt.f.f.v v16, v0");
                asm volatile("vsetvli zero, zero, e32, m8, ta, ma");
                asm volatile("vfadd.vv v8, v8, v16");
                ptr += vl; n -= (int)vl;
            }
        }
        float tmp;
        asm volatile("vsetvli zero, zero, e32, m8, ta, ma");
        asm volatile("vmv.v.i v0, 0");
        asm volatile("vfredsum.vs v0, v8, v0");
        asm volatile("vfmv.f.s %0, v0" : "=f"(tmp));
        op->d_bias[i] = tmp * inv_batchsize;
    }
    if (cycles) cycles->d_bias_cycles += get_cycle_count() - t0;

    // d_input
    t0 = get_cycle_count();
    if (conv_can_use_3x3_dx(op)) {
        conv_op_backward_input_3x3_ara(op);
    } else {
        if (ikk * oc > CONV_MAX_T_DWEIGHTS) {
            printf_("Error: conv weights transpose workspace overflow (%d)\n", ikk * oc);
            exit(1);
        }
        float *weights_T  = conv_weights_t_scratch;
        float *d_out_copy = conv_d_out_copy_scratch;
        float *d_x_col    = conv_d_x_col_scratch;
         printf_("scalar bs");
        memcpy(weights_T, op->weights, (size_t)ikk * oc * sizeof(float));
        matrix_transpose(weights_T, ikk, oc);
        for (int p = 0; p < op->batchsize; p++) {
            memcpy(d_out_copy, op->d_output + p * oc * owoh, (size_t)oc * owoh * sizeof(float));
            matrix_transpose(d_out_copy, oc, owoh);
            memset_vectorized_zero_f32(d_x_col, (size_t)ikk * (size_t)owoh);
            matrix_multiply(d_out_copy, weights_T, d_x_col, owoh, oc, ikk);
            col2img(d_x_col, op->d_input + p * op->in_units, op);
        }
    }
    if (cycles) cycles->d_input_cycles += get_cycle_count() - t0;

    op->input_col = NULL;
}

void conv_op_backward(conv_op *op)  { conv_op_backward_full(op); }
void conv_op_backward_full(conv_op *op) { conv_op_backward_full_profile(op, NULL); }

void conv_op_backward_input_only(conv_op *op)
{
    int oc   = op->out_channels;
    int ikk  = op->in_channels * op->kernel_size * op->kernel_size;
    int owoh = op->out_w * op->out_h;
    if (conv_can_use_3x3_dx(op)) {
        conv_op_backward_input_3x3_ara(op);
        op->input_col = NULL;
        return;
    }
    if (ikk * oc > CONV_MAX_T_DWEIGHTS) {
        printf_("Error: conv weights transpose workspace overflow (%d)\n", ikk * oc);
        exit(1);
    }
    float *weights_T  = conv_weights_t_scratch;
    float *d_out_copy = conv_d_out_copy_scratch;
    float *d_x_col    = conv_d_x_col_scratch;
    memcpy(weights_T, op->weights, (size_t)ikk * oc * sizeof(float));
    matrix_transpose(weights_T, ikk, oc);
    for (int p = 0; p < op->batchsize; p++) {
        memcpy(d_out_copy, op->d_output + p * oc * owoh, (size_t)oc * owoh * sizeof(float));
        matrix_transpose(d_out_copy, oc, owoh);
        memset_vectorized_zero_f32(d_x_col, (size_t)ikk * (size_t)owoh);
        matrix_multiply(d_out_copy, weights_T, d_x_col, owoh, oc, ikk);
        col2img(d_x_col, op->d_input + p * op->in_units, op);
    }
    op->input_col = NULL;
}

// =========================================================================
// Weight management
// =========================================================================
void calloc_conv_weights(conv_op *op)
{
    if (op->weights)
        memset_vectorized_zero_f32(op->weights,
            (size_t)op->out_channels * op->in_channels * op->kernel_size * op->kernel_size);
    if (op->bias)
        memset_vectorized_zero_f32(op->bias, (size_t)op->out_channels);
}

void free_conv_weights(conv_op *op) { (void)op; }

void calloc_conv_dweights(conv_op *op)
{
    if (op->d_weights)
        memset_vectorized_zero_f32(op->d_weights,
            (size_t)op->out_channels * op->in_channels * op->kernel_size * op->kernel_size);
    if (op->d_bias)
        memset_vectorized_zero_f32(op->d_bias, (size_t)op->out_channels);
}

void free_conv_dweights(conv_op *op) { (void)op; }
void save_conv_weights(conv_op *op)  { (void)op; }

void load_conv_weights(conv_op *op, float *w_array, float *b_array)
{
    size_t w_elems = (size_t)op->out_channels * op->in_channels *
                     op->kernel_size * op->kernel_size;
    memcpy(op->weights, w_array, w_elems * sizeof(float));
    memcpy(op->bias,    b_array, (size_t)op->out_channels * sizeof(float));
    if (op->weights_f16 != NULL)
        convert_f32_to_f16_vec(op->weights, op->weights_f16, w_elems);
}
