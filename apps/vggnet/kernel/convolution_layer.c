//
// File:        convolution_layer.c
// Description: Implementation of convolution layer
// Author:      Haris Wang
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
#include "utils.h"
#ifndef MIN
#define MIN(a,b) (((a) < (b)) ? (a) : (b))
#endif

// =========================================================================
// GATHER-based Vectorized img2col Support
// =========================================================================
#define MAX_GATHER_ELEMENTS (CONV_MAX_IKK * CONV_MAX_OWOH)

static uint32_t gather_offsets_buf[MAX_GATHER_ELEMENTS];
static int total_gather_elements = 0;

// memset_vectorized_zero_f32 / memcpy_vectorized_f32 now live in kernel/utils.c
// (shared by every layer); see the #include "utils.h" above.

static void pad_tensor_vectorized(float *dst, const float *src, int batch, int channels, int in_h, int in_w, int pad)
{
    int out_h = in_h + 2 * pad;
    int out_w = in_w + 2 * pad;
    size_t total = (size_t)batch * (size_t)channels * (size_t)out_h * (size_t)out_w;
    
    // ---------------------------------------------------------
    // 1. Vectorized Memset (Μηδενισμός όλου του dst)
    // ---------------------------------------------------------
    size_t n_zero = total;
    float *dst_zero = dst;
    
    // Βάζουμε 0 σε όλους τους lanes του τεράστιου καταχωρητή v8 (m8)
    // rd != x0 with rs1 = x0 -> vl = VLMAX. `vsetvli zero, zero` keeps whatever
    // vl an earlier vector op left behind, so the splat would only fill that
    // many elements and the store loop would write garbage past them.
    size_t vlmax_z;
    asm volatile("vsetvli %0, zero, e32, m8, ta, ma" : "=r"(vlmax_z));
    asm volatile("vmv.v.i v8, 0");

    while (n_zero > 0) {
        size_t vl;
        asm volatile("vsetvli %0, %1, e32, m8, ta, ma" : "=r"(vl) : "r"(n_zero));
        asm volatile("vse32.v v8, (%0)" :: "r"(dst_zero));
        dst_zero += vl;
        n_zero -= vl;
    }

    // ---------------------------------------------------------
    // 2. Vectorized Memcpy (Αντιγραφή των πραγματικών pixels)
    // ---------------------------------------------------------
    int in_plane = in_h * in_w;
    int out_plane = out_h * out_w;
    
    for (int b = 0; b < batch; b++) {
        const float *src_b = src + b * channels * in_plane;
        float *dst_b = dst + b * channels * out_plane;
        
        for (int c = 0; c < channels; c++) {
            const float *src_c = src_b + c * in_plane;
            float *dst_c = dst_b + c * out_plane + pad * out_w + pad;
            
            for (int y = 0; y < in_h; y++) {
                const float *s_ptr = src_c + y * in_w;
                float *d_ptr = dst_c + y * out_w;
                int n_copy = in_w;
                
                // Αντιγραφή της γραμμής με Vector Load -> Vector Store
                while (n_copy > 0) {
                    size_t vl;
                    asm volatile("vsetvli %0, %1, e32, m8, ta, ma" : "=r"(vl) : "r"(n_copy));
                    asm volatile("vle32.v v16, (%0)" :: "r"(s_ptr)); // Διάβασε
                    asm volatile("vse32.v v16, (%0)" :: "r"(d_ptr)); // Γράψε
                    s_ptr += vl;
                    d_ptr += vl;
                    n_copy -= vl;
                }
            }
        }
    }
}

static void unpad_tensor_vectorized(float *dst, const float *src, int batch, int channels, int in_h, int in_w, int pad)
{
    int padded_h = in_h + 2 * pad;
    int padded_w = in_w + 2 * pad;
    int out_plane = in_h * in_w;
    int in_plane = padded_h * padded_w;
    
    for (int b = 0; b < batch; b++) {
        const float *src_b = src + b * channels * in_plane;
        float *dst_b = dst + b * channels * out_plane;
        
        for (int c = 0; c < channels; c++) {
            const float *src_c = src_b + c * in_plane + pad * padded_w + pad;
            float *dst_c = dst_b + c * out_plane;
            
            for (int y = 0; y < in_h; y++) {
                const float *s_ptr = src_c + y * padded_w;
                float *d_ptr = dst_c + y * in_w;
                int n_copy = in_w;
                
                // Αφαίρεση του padding αντιγράφοντας μόνο το in_w με Vectors
                while (n_copy > 0) {
                    size_t vl;
                    asm volatile("vsetvli %0, %1, e32, m8, ta, ma" : "=r"(vl) : "r"(n_copy));
                    asm volatile("vle32.v v16, (%0)" :: "r"(s_ptr)); // Διάβασε
                    asm volatile("vse32.v v16, (%0)" :: "r"(d_ptr)); // Γράψε
                    s_ptr += vl;
                    d_ptr += vl;
                    n_copy -= vl;
                }
            }
        }
    }
}

// =========================================================================
// GATHER-based Vectorized img2col using Precomputed Byte Offsets
// =========================================================================
static void img2col_vectorized(const float *img, float *col)
{
    int n = total_gather_elements;
    const uint32_t *idx_ptr = gather_offsets_buf;
    const float *src_img = img;
    float *dst_col = col;

    while (n > 0) {
        size_t vl;
        // Ζητάμε το μέγιστο Vector Length, ομαδοποιώντας 8 καταχωρητές (m8)
        asm volatile("vsetvli %0, %1, e32, m8, ta, ma" : "=r"(vl) : "r"(n));

        // 1. Φορτώνουμε τα precomputed Byte Offsets
        asm volatile("vle32.v v8, (%0)" :: "r"(idx_ptr));

        // 2. GATHER: Μαζεύουμε τα ασύνδετα pixels από την εικόνα
        asm volatile("vluxei32.v v16, (%0), v8" :: "r"(src_img));

        // 3. STORE: Γράφουμε τα pixels απολύτως συνεχόμενα στον πίνακα col
        asm volatile("vse32.v v16, (%0)" :: "r"(dst_col));

        // Προχωράμε τους δείκτες
        dst_col += vl;
        idx_ptr += vl;
        n -= vl;
    }
}

// =========================================================================
// Precompute img2col Offsets - Called ONCE before training
// =========================================================================
// Returns the cycles spent in the actual offset generation only. The DEBUG
// printf_ calls above the timer are excluded: in Spike each character is a
// blocking HTIF syscall (~5000 cyc/char), which otherwise dwarfs the real work.
int64_t precompute_img2col_offsets_static(const conv_op *op)
{
    // The offsets index a ZERO-PADDED copy of the input image (see
    // conv_pad_input_image): rows are (in_w + 2*padding) wide and each channel
    // plane is (in_h + 2*padding) tall, so the kernel window never runs off the
    // map and no masking is needed -- the pad zeros are materialised. With
    // padding == 0 this reduces to the original conv_layer32 convention, where
    // the caller already handed in a padded map (in_w == out_w + 2).
    const int in_w_p = op->in_w + 2 * op->padding;
    const int in_h_p = op->in_h + 2 * op->padding;
    int iwih = in_w_p * in_h_p;
    int kk   = op->kernel_size * op->kernel_size; // π.χ. 9
    int ikk  = op->in_channels * kk;

    total_gather_elements = op->out_h * op->out_w * ikk;

    // NOTE: no DEBUG printf_ here. This is now called once per conv layer per
    // backward pass (the table is a single global and vggnet has four conv
    // layers with different geometry); on Spike each character is a blocking
    // HTIF syscall (~5000 cycles), which would dwarf the actual work.

    // Safety Check
    if (total_gather_elements > MAX_GATHER_ELEMENTS) {
        printf_("FATAL ERROR: Gather offsets require %d elements, but MAX_GATHER_ELEMENTS is %d\n", 
                total_gather_elements, MAX_GATHER_ELEMENTS);
        exit(1); 
    }

    int64_t _cyc0 = get_cycle_count();

    // --- Vectorized 3x3 Pattern Precomputation ---
    uint32_t patch_pattern[9]; // Υποθέτουμε max kernel 3x3
    int p_idx = 0;
    for (int j = 0; j < op->kernel_size; j++) {
        for (int i = 0; i < op->kernel_size; i++) {
            patch_pattern[p_idx++] = (i + j * in_w_p) * sizeof(float); // Bytes
        }
    }

    // Φόρτωση του Pattern μόνιμα σε έναν Vector Register (v8)
    size_t vl;
    asm volatile("vsetvli %0, %1, e32, m1, ta, ma" : "=r"(vl) : "r"(kk));
    asm volatile("vle32.v v8, (%0)" :: "r"(patch_pattern));

    uint32_t *out_ptr = gather_offsets_buf;

    // ΣΩΣΤΗ ΣΕΙΡΑ LOOPS ΓΙΑ GEMM LAYOUT: y -> x -> in_channels
    for (int st_y = 0; st_y < op->out_h * op->stride; st_y += op->stride) {
        for (int st_x = 0; st_x < op->out_w * op->stride; st_x += op->stride) {
            for (int in_c = 0; in_c < op->in_channels; in_c++) {
                
                // Υπολογισμός της "Άγκυρας" (Base Offset) σε Bytes
                uint32_t base_offset = (st_x + st_y * in_w_p + in_c * iwih) * sizeof(float);

                // v16 = v8 (pattern) + base_offset
                asm volatile("vadd.vx v16, v8, %0" :: "r"(base_offset));

                // Αποθήκευση των offsets
                asm volatile("vse32.v v16, (%0)" :: "r"(out_ptr));
                
                out_ptr += kk;
            }
        }
    }

    return get_cycle_count() - _cyc0;
}

#ifndef MIN
#define MIN(a,b) (((a) < (b)) ? (a) : (b))
#endif

static float conv1_xcol_scratch[CONV1_XCOL_ELEMS];
static float conv2_xcol_scratch[16 * 16 * (64 * 3 * 3)];
static float conv3_xcol_scratch[8 * 8 * (128 * 3 * 3)];
static float conv4_xcol_scratch[8 * 8 * (256 * 3 * 3)];
static float conv5_input_col_full[CONV5_INPUT_COL_SIZE];

static float conv_t_dweights_scratch[CONV_MAX_T_DWEIGHTS];
static float conv_d_out_copy_scratch[CONV_MAX_DOCOPY];
static float conv_d_x_col_scratch[CONV_MAX_DXCOL];
static float conv_weights_t_scratch[CONV_MAX_T_DWEIGHTS];
static float conv3x3_filter_scratch[CONV_MAX_IKK];
#define CONV_MAX_OUT_W 32
#define CONV_MAX_OUT_H 32
#define CONV_MAX_PADDED_OWOH ((CONV_MAX_OUT_W + 2) * (CONV_MAX_OUT_H + 2))
static float conv_d_out_padded_scratch[CONV_MAX_OC * CONV_MAX_PADDED_OWOH];
static float conv_d_input_tmp_scratch[CONV_MAX_OC * CONV_MAX_OWOH];

static float *conv_forward_xcol_ptr(conv_op *op, short batch_id);
static void img2col(const float *img, float *col, const conv_op *op);
static void conv_op_forward_3x3_ara(conv_op *op);
static void pad_channels(float *dst, const float *src, int channels, int in_h, int in_w, int pad);
static void pack_conv3x3_filter_rot180_dx(const conv_op *op, int ic, float *dst);
static int conv_can_use_3x3_dx(const conv_op *op);
static void conv_op_backward_input_3x3_ara(conv_op *op);

// =========================================================================
// Verification: Compare scalar vs vectorized img2col implementations
// =========================================================================
void verify_img2col_implementations(const conv_op *op, const float *test_input)
{
    // Use existing scratch buffers (no malloc - bare metal only!)
    float *col_scalar = conv1_xcol_scratch;
    float *col_vectorized = conv2_xcol_scratch;
    
    printf_("\n========== img2col Implementation Verification ==========\n");
    printf_("Running both implementations with same test input...\n");
    
    // Run scalar implementation
    img2col(test_input, col_scalar, op);
    
    // Run vectorized GATHER implementation
    img2col_vectorized(test_input, col_vectorized);
    
    // Calculate total elements in output
    int total_elements = op->out_h * op->out_w * op->in_channels * op->kernel_size * op->kernel_size;
    
    // Compare outputs element-by-element
    int diffs = 0;
    float max_diff = 0.0f;
    float sum_diff = 0.0f;
    
    for (int i = 0; i < total_elements; i++) {
        float scalar_val = col_scalar[i];
        float vect_val = col_vectorized[i];
        float diff = fabsf(scalar_val - vect_val);
        
        if (diff > 1e-6f) {
            diffs++;
            sum_diff += diff;
            if (diff > max_diff) max_diff = diff;
            
            // Print first 10 differences
            if (diffs <= 10) {
                printf_("  Diff at [%d]: scalar=%.6f, vectorized=%.6f, diff=%.6e\n",
                        i, scalar_val, vect_val, diff);
            }
        }
    }
    
    // Print summary
    if (diffs == 0) {
        printf_("✓ VERIFICATION PASSED: Both implementations produce identical results!\n");
        printf_("  Total elements compared: %d\n", total_elements);
    } else {
        printf_("✗ VERIFICATION FAILED: Found %d differences\n", diffs);
        printf_("  Total elements: %d, Differences: %d (%.2f%%)\n", 
                total_elements, diffs, (100.0f * diffs) / total_elements);
        printf_("  Max difference: %.6e\n", max_diff);
        printf_("  Average difference: %.6e\n", sum_diff / diffs);
        if (diffs > 10) {
            printf_("  (showing first 10 of %d differences)\n", diffs);
        }
    }
    printf_("==========================================================\n\n");
}

// Vectorized: the scalar loop read weights[idx*OC + oc] for idx = 0..n-1, which
// is exactly a strided load from (weights + oc) with stride OC*sizeof(float).
// Mirrors conv_layer's pack_conv3x3_filter. Was ~94 cycles scalar.
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

// Vectorized along the KERNEL axis (kk elements), not out_channels.
//
// Scalar form: dst[oc*kk + j] = weights[(ic*kk + flip(j))*OC + oc], and for a
// square kernel flip(j) = (k-1-ky)*k + (k-1-kx) = kk-1-j. Substituting:
//     dst[oc*kk + j] = weights[(ic*kk + kk-1)*OC + oc - j*OC]
// i.e. for a fixed oc the source walks BACKWARDS with byte stride -OC*4, which
// vlse32 handles directly (RVV strides are signed). So one strided load of kk
// elements + one unit-stride store per oc.
//
// Vectorizing over out_channels instead would run at vl=1 when OC==1 (the common
// case here) and is slower than the scalar loop -- measured 473 -> 573 cycles.
static void pack_conv3x3_filter_rot180_dx(const conv_op *op, int ic, float *dst)
{
    int k  = op->kernel_size;
    int kk = k * k;
    int OC = op->out_channels;
    int stride_bytes = -OC * (int)sizeof(float);   // negative: walk the kernel backwards

    for (int oc = 0; oc < OC; oc++) {
        const float *src = op->weights + (ic * kk + kk - 1) * OC + oc;
        float *d = dst + oc * kk;
        int n = kk;
        while (n > 0) {
            size_t vl;
            asm volatile("vsetvli %0, %1, e32, m8, ta, ma" : "=r"(vl) : "r"(n));
            asm volatile("vlse32.v v8, (%0), %1" :: "r"(src), "r"(stride_bytes));
            asm volatile("vse32.v v8, (%0)" :: "r"(d));
            src -= (int)vl * OC;
            d   += vl;
            n   -= (int)vl;
        }
    }
}

static int conv_can_use_3x3_ara(const conv_op *op)
{
    if (op->kernel_size != 3 || op->stride != 1)
        return 0;
    if (op->in_w != op->out_w + 2 || op->in_h != op->out_h + 2)
        return 0;
    return 1;
}

static int conv_can_use_3x3_dx(const conv_op *op)
{
    if (op->kernel_size != 3 || op->stride != 1)
        return 0;
    // Two supported layouts:
    //   conv_layer32: inputs stored PRE-PADDED      (in == out + 2)
    //   vggnet:       SAME conv over UNPADDED maps  (in == out, padding == 1)
    // Without the second case vggnet fell through to the scalar col2img
    // fallback, which is both slow and written for the pre-padded layout.
    const int prepadded = (op->in_w == op->out_w + 2 && op->in_h == op->out_h + 2);
    const int same_unpadded = (op->in_w == op->out_w && op->in_h == op->out_h &&
                               op->padding == (op->kernel_size - 1) / 2);
    if (!prepadded && !same_unpadded)
        return 0;
    if ((size_t)op->out_channels * 9 > (size_t)CONV_MAX_IKK)
        return 0;
    return 1;
}

// Forward padding scratch: fconv3d_CHx3x3_f32 reads each input channel as a
// zero-padded (in_h+2p) x (in_w+2p) plane. vggnet stores feature maps UNPADDED
// and chains conv -> bn -> conv, so we pad each image here per layer. (The
// conv_layer32 microbenchmark fed an already-padded buffer, so its forward had
// no pad step.) Sized for the widest layer: CONV_MAX_OC channels x padded 32x32.
static float conv_fwd_pad_scratch[CONV_MAX_OC * CONV_MAX_PADDED_OWOH];

static void conv_op_forward_3x3_ara(conv_op *op)
{
    int out_plane = op->out_w * op->out_h;
    int in_channels = op->in_channels;
    int out_channels = op->out_channels;

    for (int b = 0; b < op->batchsize; b++) {
        const float *input_raw = op->input + b * op->in_units;
        pad_channels(conv_fwd_pad_scratch, input_raw, in_channels,
                     op->in_h, op->in_w, op->padding);
        const float *input_b = conv_fwd_pad_scratch;
        float *output_b = op->output + b * op->out_units;

        for (int oc = 0; oc < out_channels; oc++) {
            pack_conv3x3_filter(op, oc, conv3x3_filter_scratch);
            float *out_oc = output_b + oc * out_plane;
            fconv3d_CHx3x3_f32(out_oc, input_b, conv3x3_filter_scratch,
                               op->out_h, op->out_w, in_channels, op->bias[oc]);
        }

        // DISABLED: dead work that made the forward look ~11x slower than the
        // FP16 forward. This ran the SCALAR img2col (~15k instrs for 64x9=576
        // elements) inside the timed forward, but conv_op_backward_full_profile
        // re-sets input_col and recomputes x_col with the VECTORIZED
        // img2col_vectorized() before anything reads it -- so the result here was
        // always overwritten. conv_layer's FP16 forward has no such block, which
        // is why forward was 1440 (f16) vs 16728 (f32) cycles at W=64.
        // if (op->input_col != NULL) {
        //     float *x_col = conv_forward_xcol_ptr(op, (short)b);
        //     img2col(input_b, x_col, op);
        // }
    }
}

static void pad_channels(float *dst, const float *src, int channels, int in_h, int in_w, int pad)
{
    int out_h = in_h + 2 * pad;
    int out_w = in_w + 2 * pad;
    size_t total = (size_t)channels * (size_t)out_h * (size_t)out_w;
    memset_vectorized_zero_f32(dst, total);

    int in_plane = in_h * in_w;
    int out_plane = out_h * out_w;
    for (int c = 0; c < channels; c++) {
        const float *src_c = src + c * in_plane;
        float *dst_c = dst + c * out_plane + pad * out_w + pad;
        for (int y = 0; y < in_h; y++) {
            // Vectorized (elements, not bytes). libc memcpy needs dest|src|len ALL
            // 8-byte aligned or it drops to a byte-at-a-time loop -- and dst_c is
            // offset by (pad*out_w + pad) floats, an odd float offset when pad=1,
            // so it took the byte path on every row. This mirrors pad_channels_f16.
            memcpy_vectorized_f32(dst_c + y * out_w, src_c + y * in_w, (size_t)in_w);
        }
    }
}

static void conv_op_backward_input_3x3_ara(conv_op *op)
{
    // Padding applied to d_output before the full (rot180) convolution:
    //   conv_layer32 pre-padded layout (in == out + 2) -> (in_w - out_w)/2 == 1
    //   vggnet SAME conv over unpadded maps (in == out) -> (kernel_size - 1)/2
    int pad = (op->in_w > op->out_w) ? (op->in_w - op->out_w) / 2
                                     : (op->kernel_size - 1) / 2;
    int out_plane = op->out_w * op->out_h;
    int padded_plane = (op->out_h + 2 * pad) * (op->out_w + 2 * pad);

    if (pad <= 0) {
        printf_("Error: invalid padding for 3x3 d_input\n");
        exit(1);
    }
    if ((size_t)op->out_channels * (size_t)padded_plane > (size_t)CONV_MAX_OC * (size_t)CONV_MAX_PADDED_OWOH) {
        printf_("Error: d_output padded workspace overflow\n");
        exit(1);
    }
    if ((size_t)op->in_channels * (size_t)out_plane > (size_t)CONV_MAX_OC * (size_t)CONV_MAX_OWOH) {
        printf_("Error: d_input workspace overflow\n");
        exit(1);
    }

    for (int b = 0; b < op->batchsize; b++) {
        const float *d_out_b = op->d_output + b * op->out_units;
        float *d_out_padded = conv_d_out_padded_scratch;
        pad_channels(d_out_padded, d_out_b, op->out_channels, op->out_h, op->out_w, pad);

        float *d_input_tmp = conv_d_input_tmp_scratch;
        for (int ic = 0; ic < op->in_channels; ic++) {
            pack_conv3x3_filter_rot180_dx(op, ic, conv3x3_filter_scratch);
            float *d_in_ch = d_input_tmp + ic * out_plane;
            fconv3d_CHx3x3_f32(d_in_ch, d_out_padded, conv3x3_filter_scratch,
                               op->out_h, op->out_w, op->out_channels, 0.0f);
        }

        float *d_in_b = op->d_input + b * op->in_units;
        if (op->in_w == op->out_w && op->in_h == op->out_h) {
            // vggnet: SAME conv over UNPADDED storage (in == out), so the
            // full-convolution result above already IS d_input at the input's
            // spatial size -- copy it straight through.
            //
            // conv_layer32 instead kept inputs pre-padded (in_w == out_w + 2)
            // and re-padded dx here to match. Doing that in vggnet writes
            // in_channels*(out_h+2)*(out_w+2) into a buffer holding only
            // in_units = in_channels*out_h*out_w (e.g. conv4: 24*18*18 = 7776
            // into 6144 -> 1632 floats of overflow per image) AND hands the
            // previous layer a padded map it reads as unpadded.
            memcpy_vectorized_f32(d_in_b, d_input_tmp,
                                  (size_t)op->in_channels * (size_t)out_plane);
        } else {
            pad_channels(d_in_b, d_input_tmp, op->in_channels, op->out_h, op->out_w, pad);
        }
    }
}

static float *conv_forward_xcol_ptr(conv_op *op, short batch_id)
{
    int col_size_per_image = (op->in_channels * op->kernel_size * op->kernel_size) * (op->out_w * op->out_h);
    if (op->layer_id == 5)
        return op->input_col + batch_id * col_size_per_image;

    // Only one image's x_col is live at a time (both the forward and the
    // d_weights loop consume it before moving on), so these are per-layer
    // single-image scratches, not batch-sized.
    static const int xcol_capacity[5] = {
        0, CONV1_XCOL_ELEMS, 16 * 16 * (64 * 3 * 3),
        8 * 8 * (128 * 3 * 3), 8 * 8 * (256 * 3 * 3)
    };
    if (op->layer_id >= 1 && op->layer_id <= 4 &&
        col_size_per_image > xcol_capacity[op->layer_id]) {
        printf_("Error: conv%d x_col needs %d elems, scratch holds %d\n",
                op->layer_id, col_size_per_image, xcol_capacity[op->layer_id]);
        exit(1);
    }

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


// Zero-pad one image into conv_fwd_pad_scratch and return it. Both img2col
// flavours and fconv3d consume the SAME padded layout, so the im2col path and
// the sliding path cannot drift apart in convention again.
static const float *conv_pad_input_image(const conv_op *op, const float *img_raw)
{
    pad_channels(conv_fwd_pad_scratch, img_raw, op->in_channels,
                 op->in_h, op->in_w, op->padding);
    return conv_fwd_pad_scratch;
}

// `img` must be the PADDED image from conv_pad_input_image() -- rows are
// (in_w + 2*padding) wide -- which is what the gather offsets built by
// precompute_img2col_offsets_static() assume too.
static void img2col(const float *img, float *col, const conv_op *op) // not classic. row-major patches
{
    const int in_w_p = op->in_w + 2 * op->padding;
    const int in_h_p = op->in_h + 2 * op->padding;
    int iwih = in_w_p * in_h_p;
    int kk   = op->kernel_size * op->kernel_size; //number of pixels in a channel of a kernel
    int ikk  = op->in_channels * kk; //total number of pixels in a kernel

    //st_x,y == sum of stride x,y


    for (int in_c = 0; in_c < op->in_channels; in_c++)
    {
        int out_y = 0;
        for (int st_y = 0; st_y < op->out_h * op->stride; st_y += op->stride, out_y++)
        {
            int out_x = 0;
            for (int st_x = 0; st_x < op->out_w * op->stride; st_x += op->stride, out_x++) //we move per stride
            {

                int patch_idx = out_y * op->out_w + out_x; // row-major patch index
                int x_col_offset = patch_idx * ikk + in_c * kk; //+0
                // in_c*kk==prosdiorismos pixel se sygkekrimeno channel
                //patch_idx*ikk=poy ksekinaei h nea grammh pou antistoixei
                //so in those 2 lines we set which is the patch of the image, we want to look at

                for (int j = 0; j < op->kernel_size; j++)
                {
                    for (int i = 0; i < op->kernel_size; i++)
                    {
                        int input_offset = (st_x + i) + (st_y + j) * in_w_p + in_c * iwih;
                        col[x_col_offset] = img[input_offset];
                        x_col_offset++;
                    }
                }
            }
        }
    }
} //so we destroy data locality in this algorithm in order to have data locality in GEMM


static void conv_op_forward_single(conv_op *op, short batch_id)
{
    float *x_col = conv_forward_xcol_ptr(op, batch_id);
    // vggnet stores feature maps UNPADDED, so pad before im2col -- otherwise this
    // path computes a shifted, uncentred convolution and disagrees with the
    // fconv3d forward (which pads via the same helper).
    const float *t_input = conv_pad_input_image(op, op->input + batch_id * op->in_units);
    float *t_output = op->output + batch_id * op->out_units;
    int ikk  = op->in_channels * op->kernel_size * op->kernel_size;
    int owoh = op->out_w * op->out_h;
    // 
    // >>>>>>>shape<<<<<<<
    //  
    // t_input    [ic,ih,iw]
    // x_col      [owoh,ikk]
    // weights    [ikk,oc]
    // t_output   [oc,oh,ow]
    // >>>>>>>>>>>>>>>>>>>
    //
    img2col(t_input, x_col, op);
    matrix_multiply(x_col, op->weights, t_output, owoh, ikk, op->out_channels);
    matrix_transpose(t_output, owoh, op->out_channels);

    register int o_offset=0;
    for (int i = 0; i < op->out_channels; i++)
    {
        register float tmp = op->bias[i];
        while (o_offset < (i+1)*owoh)
        {
            t_output[o_offset++] += tmp;
        }
    }


    return;
}

/*

typedef struct conv_args{
    conv_op *op;
    short batch_id;
    short st_tunits;
    short ed_tunits;
} conv_args;
*/


void conv_op_forward(conv_op *op)
{
    /**
     * conv2d forward
     * 
     * Input:
     *      op->input
     *      op->weights
     *      op->bias
     * Output:
     *      op->output
     * */
    if (op->layer_id == 5) {
        if (op->batchsize > ALEXNET_STATIC_MAX_BATCH) {
            printf_("Error: conv5 batchsize %d exceeds static max %d\n", op->batchsize, ALEXNET_STATIC_MAX_BATCH);
            exit(1);
        }
        op->input_col = conv5_input_col_full;
        // NOTE: element count, NOT bytes -- do not re-add * sizeof(float).
        // The libc memset in common/string.c drops to a BYTE loop unless dest is
        // 8-byte aligned; conv5_input_col_full is only 4-byte aligned, so zeroing
        // 2304 bytes cost 9249 cycles and dominated the entire forward pass
        // (the conv kernel itself is 128). vse32 needs only 4-byte alignment, so
        // this has no such cliff.
        memset_vectorized_zero_f32(op->input_col,
               (size_t)op->batchsize * (size_t)(op->in_channels * op->kernel_size * op->kernel_size) *
               (size_t)(op->out_w * op->out_h));
    } else {
        op->input_col = NULL;
    }

#ifdef USE_IM2COL_CONV
    // Diagnostic: route the forward through the scalar im2col + matrix_multiply
    // reference instead of the vectorized fconv3d path, to bisect vector-vs-scalar
    // discrepancies. (conv_op_forward_im2col loops over the batch itself.)
    conv_op_forward_im2col(op);
#else
    conv_op_forward_3x3_ara(op);
#endif
    return;


}

void conv_op_forward_im2col(conv_op *op)
{
    if (op->layer_id == 5 && op->input_col == NULL) {
        printf_("Error: conv_op_forward_im2col requires input_col for layer_id=5\n");
        exit(1);
    }

    for (int p = 0; p < op->batchsize; p++)
    {
        conv_op_forward_single(op, (short)p);
    }
}


static void col2img(const float *col, float *img, const conv_op *op)
{
    int iwih = op->in_w * op->in_h;
    int kk   = op->kernel_size * op->kernel_size;
    int ikk  = op->in_channels * kk;

    int out_y = 0;
    for (int st_y = 0; st_y < op->out_h * op->stride; st_y += op->stride, out_y++)
    {
        int out_x = 0;
        for (int st_x = 0; st_x < op->out_w * op->stride; st_x += op->stride, out_x++)
        {
            for (int in_c = 0; in_c < op->in_channels; in_c++)
            {
                // Ο ίδιος ασφαλής υπολογισμός
                int patch_idx = out_y * op->out_w + out_x;
                int x_col_offset = patch_idx * ikk + in_c * kk;

                for (int j = 0; j < op->kernel_size; j++)
                {
                    for (int i = 0; i < op->kernel_size; i++)
                    {
                        // img is the UNPADDED d_input map, so undo the centred
                        // padding and drop taps that fall outside it. (The old
                        // form indexed st_x + i up to out_w + 1 with a row
                        // stride of in_w == out_w, scattering off the end of
                        // every row and past the end of the plane.)
                        int ix = st_x + i - op->padding;
                        int iy = st_y + j - op->padding;
                        if (ix >= 0 && ix < op->in_w && iy >= 0 && iy < op->in_h)
                            img[ix + iy * op->in_w + in_c * iwih] += col[x_col_offset];
                        x_col_offset++;
                    }
                }
            }
        }
    }
}


void conv_op_backward(conv_op *op)
{
    conv_op_backward_full(op);
}

void conv_op_backward_full(conv_op *op)
{
    conv_op_backward_full_profile(op, NULL);
}

void conv_op_backward_full_profile(conv_op *op, conv_backward_cycle_breakdown *cycles)
{
    int oc = op->out_channels;
    int ikk = op->in_channels * op->kernel_size * op->kernel_size;
    int owoh = op->out_w * op->out_h;
    int64_t t0 = 0;

    if (cycles) {
        cycles->d_input_cycles = 0;
        cycles->d_bias_cycles = 0;
        cycles->d_weights_im2col_cycles = 0;
        cycles->d_weights_cycles = 0;
    }

    float *t_d_weights = conv_t_dweights_scratch;
    int64_t dweights_total_t0 = get_cycle_count();

    if (op->layer_id == 5)
        op->input_col = conv5_input_col_full;

    // The gather-offset table is a single global, and vggnet has four conv
    // layers with different geometry, so rebuild it for THIS layer before the
    // im2col below. (In conv_layer32 there was one layer and one precompute at
    // startup.) Cost is out_h*out_w*in_channels tiny vector stores -- negligible
    // next to the GEMM it feeds.
    precompute_img2col_offsets_static(op);

    for (int p = 0; p < op->batchsize; p++)
    {
        t0 = get_cycle_count();
        // Re-ENABLED (was disabled in conv_layer32: that app benchmarked the conv
        // kernel and deliberately left d_weights at zero). vggnet actually trains
        // the conv stack, so without this the filters never update.
        //
        // x_col[owoh][ikk] is built from the PADDED image, matching the offsets
        // precomputed above and the fconv3d forward.
        float *x_col = conv_forward_xcol_ptr(op, (short)p);
        const float *img_p = conv_pad_input_image(op, op->input + p * op->in_units);
        if (total_gather_elements > 0) {
            img2col_vectorized(img_p, x_col);
        } else {
            printf_("not vectorized im2col");
            img2col(img_p, x_col, op);
        }

        if (cycles)
            cycles->d_weights_im2col_cycles += get_cycle_count() - t0;

        // No memset needed: matrix_multiply_32 now OVERWRITES c on every exit
        // path (see the contract note on its definition). It used to accumulate
        // in the M-padding branch, which every conv here takes (M = oc is
        // 12 -> 16 or 24 -> 32), so this loop required a zeroing pass to avoid
        // piling all 32 batch images on top of each other.
        t0 = get_cycle_count();
        // t_d_weights[oc][ikk] = d_output[oc][owoh] * x_col[owoh][ikk].
        // N=owoh<=1024 and K=ikk<=216 are well inside FMATMUL_MAX_N/K and
        // padded_m <= FMATMUL_MAX_M, so this stays on the vectorized fmatmul_32
        // path rather than falling back to the scalar reference.
        matrix_multiply_32(op->d_output + p * oc * owoh, x_col, t_d_weights,
                           oc, owoh, ikk);

        int stride_bytes = oc * (int)sizeof(float);
        for (int j = 0; j < oc; j++) {
            float *src = t_d_weights + j * ikk;
            float *dst = op->d_weights + j;
            int n = ikk;
            while (n > 0) {
                size_t vl;
                asm volatile("vsetvli %0, %1, e32, m8, ta, ma" : "=r"(vl) : "r"(n));
                asm volatile("vle32.v v8, (%0)" :: "r"(src));
                asm volatile("vlse32.v v16, (%0), %1" :: "r"(dst), "r"(stride_bytes));
                asm volatile("vfadd.vv v16, v16, v8");
                asm volatile("vsse32.v v16, (%0), %1" :: "r"(dst), "r"(stride_bytes));
                src += vl;
                dst += vl * oc;
                n -= (int)vl;
            }
        }
    }

    if (cycles)
        cycles->d_weights_cycles += get_cycle_count() - dweights_total_t0;
    

    // No 1/batchsize pass here: d_output already carries it, applied once in
    // cross_entropy_loss(). See the comment there.

    // Bias υπολογισμός
    // for (int i = 0; i < op->out_channels; i++)
    // {
    //     float tmp = 0.0f;
    //     for (int p = 0; p < op->batchsize; p++)
    //         for (int s = i * owoh; s < (i + 1) * owoh; s++)
    //             tmp += op->d_output[p * oc * owoh + s];
    //     op->d_bias[i] = tmp / op->batchsize;
    // }

    // int oc = op->out_channels;
    // int owoh = op->out_w * op->out_h;
    t0 = get_cycle_count();

    for (int i = 0; i < oc; i++)
    {
        size_t vlmax_a;
        asm volatile("vsetvli %0, zero, e32, m8, ta, ma" : "=r"(vlmax_a));
        asm volatile("vmv.v.i v8, 0");

        for (int p = 0; p < op->batchsize; p++)
        {
            const float *ptr = op->d_output + p * oc * owoh + i * owoh;
            int n = owoh;
            while (n > 0) {
                size_t vl;
                asm volatile("vsetvli %0, %1, e32, m8, ta, ma" : "=r"(vl) : "r"(n));
                asm volatile("vle32.v v16, (%0)" :: "r"(ptr));
                asm volatile("vfadd.vv v8, v8, v16");
                ptr += vl;
                n -= vl;
            }
        }

        float tmp;
        size_t vlmax_r;
        asm volatile("vsetvli %0, zero, e32, m8, ta, ma" : "=r"(vlmax_r));
        asm volatile("vmv.v.i v0, 0");
        asm volatile("vfredsum.vs v0, v8, v0");
        asm volatile("vfmv.f.s %0, v0" : "=f"(tmp));

        op->d_bias[i] = tmp;   // 1/batchsize already folded into d_output
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
        float *weights_T = conv_weights_t_scratch;
        memcpy(weights_T, op->weights, ikk * oc * sizeof(float));
        matrix_transpose(weights_T, ikk, oc);

        float *d_out_copy = conv_d_out_copy_scratch;
        float *d_x_col = conv_d_x_col_scratch;

        for (int p = 0; p < op->batchsize; p++)
        {
            memcpy(d_out_copy, op->d_output + p * oc * owoh, oc * owoh * sizeof(float));
            matrix_transpose(d_out_copy, oc, owoh);
            memset_vectorized_zero_f32(d_x_col, (size_t)ikk * (size_t)owoh);
            matrix_multiply_eps(d_out_copy, weights_T, d_x_col, owoh, oc, ikk,
                                0.00001f / (float)op->batchsize);
            col2img(d_x_col, op->d_input + p * op->in_units, op);
        }
    }
    if (cycles) cycles->d_input_cycles += get_cycle_count() - t0;

    op->input_col = NULL;
}

void conv_op_backward_input_only(conv_op *op)
{
    // Only propagate d_input for frozen convolution layers.
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
    float *weights_T = conv_weights_t_scratch;
    memcpy(weights_T, op->weights, ikk * oc * sizeof(float));
    matrix_transpose(weights_T, ikk, oc);

    float *d_out_copy = conv_d_out_copy_scratch;
    float *d_x_col = conv_d_x_col_scratch;

    for (int p = 0; p < op->batchsize; p++)
    {
        memcpy(d_out_copy, op->d_output + p * oc * owoh, oc * owoh * sizeof(float));
        matrix_transpose(d_out_copy, oc, owoh);
        memset_vectorized_zero_f32(d_x_col, (size_t)ikk * (size_t)owoh);
        matrix_multiply_eps(d_out_copy, weights_T, d_x_col, owoh, oc, ikk,
                            0.00001f / (float)op->batchsize);
        col2img(d_x_col, op->d_input + p * op->in_units, op);
    }

    op->input_col = NULL;
}

void calloc_conv_weights(conv_op *op)
{
    if (op->weights)
        memset_vectorized_zero_f32(op->weights, (size_t)op->out_channels * op->in_channels * op->kernel_size * op->kernel_size);
    if (op->bias)
        memset_vectorized_zero_f32(op->bias, (size_t)op->out_channels);
}

void free_conv_weights(conv_op *op)
{
    (void)op;
}

void calloc_conv_dweights(conv_op *op)
{
    if (op->d_weights)
        memset_vectorized_zero_f32(op->d_weights, (size_t)op->out_channels * op->in_channels * op->kernel_size * op->kernel_size);
    if (op->d_bias)
        memset_vectorized_zero_f32(op->d_bias, (size_t)op->out_channels);
}

void free_conv_dweights(conv_op *op)
{
    (void)op;
}

void save_conv_weights(conv_op *op)
{
    (void)op;
}


void load_conv_weights(conv_op *op, float *w_array, float *b_array)
{
    memcpy(op->weights, w_array,
           sizeof(float) * op->out_channels * op->in_channels * op->kernel_size * op->kernel_size);
    memcpy(op->bias, b_array, sizeof(float) * op->out_channels);
}
