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
#ifndef MIN
#define MIN(a,b) (((a) < (b)) ? (a) : (b))
#endif

// =========================================================================
// GATHER-based Vectorized img2col Support
// =========================================================================
#define MAX_GATHER_ELEMENTS (CONV_MAX_IKK * CONV_MAX_OWOH)

static uint32_t gather_offsets_buf[MAX_GATHER_ELEMENTS];
static int total_gather_elements = 0;

static inline void memset_vectorized_zero_f32(float *dst, size_t n_elements) {
    // 1. Προετοιμασία: Γεμίζουμε 8 Vector Registers (m8) με μηδενικά
    // `vsetvli zero, zero` (rd = x0 AND rs1 = x0) KEEPS the caller's vl and only
    // changes vtype -- it does NOT select VLMAX. The splat then fills just those
    // lanes while the drain loop below stores at vl = min(n, VLMAX), writing the
    // untouched lanes out as GARBAGE instead of zeros. rd != x0 with rs1 = x0
    // requests AVL = ~0 -> vl = VLMAX, so the whole register group is zeroed.
    size_t vlmax_z;
    asm volatile("vsetvli %0, zero, e32, m8, ta, ma" : "=r"(vlmax_z));
    asm volatile("vmv.v.i v16, 0");

    // 2. Εκτέλεση: Αδειάζουμε τα μηδενικά στη μνήμη με μέγιστο bandwidth
    while (n_elements > 0) {
        size_t vl;
        asm volatile("vsetvli %0, %1, e32, m8, ta, ma" : "=r"(vl) : "r"(n_elements));
        asm volatile("vse32.v v16, (%0)" :: "r"(dst));
        
        dst += vl;
        n_elements -= vl;
    }
}

static inline void memcpy_vectorized_f32(float *dst, const float *src, size_t n_elements) {
    while (n_elements > 0) {
        size_t vl;
        // Χρησιμοποιούμε m8 για να δεσμεύσουμε 8 Vector Registers μαζί!
        asm volatile("vsetvli %0, %1, e32, m8, ta, ma" : "=r"(vl) : "r"(n_elements));
        
        // Φόρτωση και Αποθήκευση στη μέγιστη ταχύτητα του διαύλου μνήμης
        asm volatile("vle32.v v16, (%0)" :: "r"(src));
        asm volatile("vse32.v v16, (%0)" :: "r"(dst));
        
        src += vl;
        dst += vl;
        n_elements -= vl;
    }
}

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
    // `vsetvli zero, zero` (rd = x0 AND rs1 = x0) KEEPS the caller's vl and only
    // changes vtype -- it does NOT select VLMAX. The splat then fills just those
    // lanes while the drain loop below stores at vl = min(n, VLMAX), writing the
    // untouched lanes out as GARBAGE instead of zeros. rd != x0 with rs1 = x0
    // requests AVL = ~0 -> vl = VLMAX, so the whole register group is zeroed.
    size_t vlmax_p;
    asm volatile("vsetvli %0, zero, e32, m8, ta, ma" : "=r"(vlmax_p));
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
    int iwih = op->in_w * op->in_h;
    int kk   = op->kernel_size * op->kernel_size; // π.χ. 9
    int ikk  = op->in_channels * kk;
    
    total_gather_elements = op->out_h * op->out_w * ikk;

    printf_("DEBUG: precompute_img2col_offsets_static called\n");
    printf_("  out_h=%d, out_w=%d, in_channels=%d, kk=%d, ikk=%d\n", 
            op->out_h, op->out_w, op->in_channels, kk, ikk);
    printf_("  total_gather_elements=%d, MAX_GATHER_ELEMENTS=%d\n", 
            total_gather_elements, MAX_GATHER_ELEMENTS);

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
            patch_pattern[p_idx++] = (i + j * op->in_w) * sizeof(float); // Bytes
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
                uint32_t base_offset = (st_x + st_y * op->in_w + in_c * iwih) * sizeof(float);

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
    // Two layouts are supported:
    //   PRE-PADDED  (in == out + 2): what alexnet_train() switches conv1 to, so
    //               the kernel consumes train_input_pad_buf directly.
    //   SAME/UNPADDED (in == out, padding == (k-1)/2): the logical shape
    //               setup_alexnet() leaves in place, and what a network that
    //               chains conv -> bn -> conv (e.g. vggnet) actually stores.
    // Only the first was accepted before, so under the logical geometry this
    // returned 0 and the backward silently fell through to col2img -- which is
    // itself written for the pre-padded layout and scatters off the end of every
    // row. Accepting both keeps the two representations interchangeable.
    const int prepadded = (op->in_w == op->out_w + 2 && op->in_h == op->out_h + 2);
    const int same_unpadded = (op->in_w == op->out_w && op->in_h == op->out_h &&
                               op->padding == (op->kernel_size - 1) / 2);
    if (!prepadded && !same_unpadded)
        return 0;
    if ((size_t)op->out_channels * 9 > (size_t)CONV_MAX_IKK)
        return 0;
    return 1;
}

static void conv_op_forward_3x3_ara(conv_op *op)
{
    int out_plane = op->out_w * op->out_h;
    int in_channels = op->in_channels;
    int out_channels = op->out_channels;

    for (int b = 0; b < op->batchsize; b++) {
        const float *input_b = op->input + b * op->in_units;
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
    //   pre-padded layout (in == out + 2) -> (in_w - out_w)/2 == 1
    //   SAME/unpadded layout (in == out)  -> (kernel_size - 1)/2
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
            // SAME/unpadded layout: the full-convolution result above already IS
            // d_input at the input's spatial size, so copy it straight through.
            // Re-padding here (correct for the pre-padded layout, where in_units
            // is itself padded) would write in_channels*(out_h+2)*(out_w+2) into
            // a buffer holding only in_channels*out_h*out_w, and hand the
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


static void img2col(const float *img, float *col, const conv_op *op) // not classic. row-major patches
{
    int iwih = op->in_w * op->in_h; 
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
                        int input_offset = (st_x + i) + (st_y + j) * op->in_w + in_c * iwih;
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
    float *t_input  = op->input + batch_id * op->in_units;
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

    conv_op_forward_3x3_ara(op);
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
                        // Undo the centred padding and drop taps outside the map.
                        // Under the PRE-PADDED layout padding == 0 and in_w ==
                        // out_w + 2, so this reduces exactly to the old
                        // (st_x + i) + (st_y + j) * in_w form. Under the
                        // SAME/unpadded layout that old form indexed st_x + i up
                        // to out_w + 1 with a row stride of in_w == out_w,
                        // scattering off the end of every row and past the plane.
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

    for (int p = 0; p < op->batchsize; p++)
    {
        t0 = get_cycle_count();
        // DISABLED: dead work. With the d_weights GEMM below commented out,
        // nothing reads x_col. Kept in lockstep with conv_layer, where the FP16
        // equivalent (img2col_vectorized_f16) is disabled because its vluxei32.v
        // gather at e16/m4 hangs on Ara RTL. Leaving this enabled charged FP32 an
        // extra ~2600 cycles that FP16 was not paying -- an unfair asymmetry.
        // Re-enable together with the GEMM.
        //
        // float *x_col = conv_forward_xcol_ptr(op, (short)p);
        // if (total_gather_elements > 0) {
        //     img2col_vectorized(op->input + p * op->in_units, x_col);
        // } else {
        //     printf_("scalar im2col (gather_elements=%d, layer_id=%d)\n", total_gather_elements, op->layer_id);
        //     img2col(op->input + p * op->in_units, x_col, op);
        // }

        if (cycles)
            cycles->d_weights_im2col_cycles += get_cycle_count() - t0;

        memset_vectorized_zero_f32(t_d_weights, (size_t)oc * (size_t)ikk);

        t0 = get_cycle_count();
//                matrix_multiply(op->d_output + p * oc * owoh, x_col, t_d_weights, oc, owoh, ikk);

        // DISABLED: benchmarking the conv kernel, not the weight-gradient GEMM.
        // t_d_weights stays zeroed by the memset above, so the scatter below
        // accumulates zeros and d_weights stays 0. Uncomment to restore.
        // matrix_multiply(op->d_output + p * oc * owoh, x_col, t_d_weights, oc, owoh, ikk);

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
    float inv_batchsize = 1.0f / (float)op->batchsize;

    for (int i = 0; i < oc; i++)
    {
        // Zero the WHOLE accumulator group. `vsetvli zero, zero` (rd = x0 AND
        // rs1 = x0) keeps whatever vl the preceding code left, so the splat
        // covered only that many lanes while the loop below accumulates into
        // min(owoh, VLMAX) of them -- any lane in between would carry garbage
        // straight into d_bias. rd != x0 with rs1 = x0 gives vl = VLMAX.
        size_t vlmax_acc;
        asm volatile("vsetvli %0, zero, e32, m8, ta, ma" : "=r"(vlmax_acc));
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
        // Reduce the WHOLE accumulator group too. Keeping the loop's trailing vl
        // here silently drops every lane above it, which for owoh > VLMAX (not a
        // multiple) loses most of the sum. The lanes the loop never touched were
        // zeroed above, so reducing at VLMAX is correct for any owoh.
        size_t vlmax_red;
        asm volatile("vsetvli %0, zero, e32, m8, ta, ma" : "=r"(vlmax_red));
        asm volatile("vmv.v.i v0, 0");
        asm volatile("vfredsum.vs v0, v8, v0");
        asm volatile("vfmv.f.s %0, v0" : "=f"(tmp));

        op->d_bias[i] = tmp * inv_batch;
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
            matrix_multiply(d_out_copy, weights_T, d_x_col, owoh, oc, ikk);
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
        matrix_multiply(d_out_copy, weights_T, d_x_col, owoh, oc, ikk);
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
