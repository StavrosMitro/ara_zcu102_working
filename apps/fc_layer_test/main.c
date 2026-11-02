//
// File:        main.c  (apps/fc_layer_test)
// Description: Mixed-precision (FP16 in/weights, FP32 accumulate) vs pure-FP32
//              equivalence test suite for the AlexNet FC layer kernels.
//
//   - The FP16 path is the *mixed precision* tree   (apps/fc_layer,   original symbol names).
//   - The FP32 path is the *reference* tree         (apps/fc_layer32, every external symbol
//                                                    suffixed with _32 so both trees link
//                                                    into one binary without collisions).
//
//   Both trees are driven on the SAME numerical inputs (FP16 path receives _Float16 casts).
//   Each kernel is exercised "one by one" and the two results are compared, then a final
//   end-to-end training-step ("in flow") test runs forward -> loss -> backward -> update on
//   both precisions and reports how far the weights diverge.
//
//   Author: Stavros Mitropoulos (test harness)
//
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#ifdef SPIKE
#include <printf.h>
#elif defined ARA_LINUX
#include <stdio.h>
#else
#include "printf.h"
#endif

// ---------------------------------------------------------------------------
//  Public interfaces of the MIXED-PRECISION (FP16) tree.
//  Quoted includes resolve to fc_layer_test/fp16/*.h (directory of this file).
//  These give us: fc_op (with _Float16 input/weights), fc_backward_cycle_breakdown,
//  matrix_multiply*/fmatmul* (FP16 operands), event_trigger (linker symbol).
// ---------------------------------------------------------------------------
#include "fp16/matrix.h"
#include "fp16/fmatmul.h"
#include "fp16/fc_layer.h"

// ---------------------------------------------------------------------------
//  Pure-FP16 tree (fc_layer16only).
//  matrix_multiply_16 / matrix_multiply_nt_16 / ... use _16 suffix so they
//  coexist with the mixed-precision symbols above without name clashes.
//  fc_op_forward_p16 is provided by the fp16pure/fc_layer_p16.c shim.
// ---------------------------------------------------------------------------
#include "fp16pure/matrix.h"
#include "fp16pure/fc_layer_p16.h"

// ---------------------------------------------------------------------------
//  Interfaces of the FP32 reference tree, declared BY HAND.
//  We do NOT #include the fp32 headers: they reuse the same include guards
//  (FMATMUL_H, ...) and the same struct tag (fc_op) as the FP16 headers, so
//  including both would clash. Instead we mirror the _32 symbols here. The
//  fp32 fc_op layout is identical except every pointer is `float *`.
// ---------------------------------------------------------------------------
typedef struct fc_op_32 {
    float *input;   float *d_input;
    float *output;  float *d_output;
    float *weights; float *d_weights;
    float *bias;    float *d_bias;
    int in_units, out_units;
    short batchsize;
    short layer_id;
} fc_op_32;

typedef struct fc_bw_cyc_32 {
    int64_t d_input_cycles;
    int64_t d_bias_cycles;
    int64_t d_weights_cycles;
} fc_bw_cyc_32;

void matrix_multiply_32(const float *a, const float *b, float *c, const int M, const int N, const int K);
void matrix_multiply_fused_32(const float *a, const float *b, const float *bias, float *c, const int M, const int N, const int K);
void matrix_multiply_nt_32(const float *a, const float *b, float *c, const int M, const int N, const int K);
void matrix_multiply_tn_32(const float *a, const float *b, float *c, const int M, const int N, const int K);
void matrix_multiply_nt_deferred_32(const float *a, const float *b, float *c, const int M, const int N, const int K);
void fc_op_forward_32(fc_op_32 *op);
void fc_op_backward_full_profile_32(fc_op_32 *op, fc_bw_cyc_32 *cycles);

// ---------------------------------------------------------------------------
//  Dataset, straight from data.S (.incbin of generated_data/*.bin), FP32.
// ---------------------------------------------------------------------------
extern const float test_inputs[];     // [>= BATCH * IN]  (32768 bytes = 8192 floats = 4x2048)
extern const float test_targets[];    // [>= BATCH * OUT] ( 8192 bytes = 2048 floats = 4x512)
extern const int   test_labels[];     // [>= BATCH]

// ---------------------------------------------------------------------------
//  Bare-metal / Spike runtime glue (normally provided by each tree's main.c,
//  which we exclude). Harmless if unreferenced.
// ---------------------------------------------------------------------------
void __libc_init_array(void) {}
void __libc_fini_array(void) {}

#ifdef SPIKE
extern volatile uint64_t tohost;
uintptr_t handle_trap(uintptr_t cause, uintptr_t epc, uintptr_t regs[32])
{
    uintptr_t mtval = 0;
    asm volatile("csrr %0, mtval" : "=r"(mtval));
    printf_("\n!!! FATAL EXCEPTION: cause=%lu epc=0x%lx mtval=0x%lx !!!\n", cause, epc, mtval);
    tohost = ((0x100 + (cause & 0x3ff)) << 1) | 1;
    while (1) {}
}
#endif

// ===========================================================================
//  Problem dimensions (match the real FC1 layer / data.S vectors).
//  BATCH = 2 is mandatory: calc_bias_gradient_vec_batch2 is hardcoded for 2 rows.
// ===========================================================================
#define IN    2048
#define OUT   512
#define BATCH 2

// ---- copied loss/optimizer configuration (must match train.c) -------------
#define LEARNING_RATE 0.00001f
#define ALEXNET_USE_MOMENTUM 1
#define MAX(a,b) (((a) > (b)) ? (a) : (b))
#define MIN(a,b) (((a) < (b)) ? (a) : (b))

// Comparison tolerance for "PASS". FP16 inputs rounded to ~2^-11 relative, then
// accumulated in FP32, so a few percent worst-case relative error is expected.
#define REL_TOL 3.0e-2
#define ABS_FLOOR 1.0e-4   // ignore relative error of near-zero reference values

// ===========================================================================
//  Static buffers (.bss). L2 is 256 MB, so full FC dims are fine.
// ===========================================================================
static float    A_f32[BATCH * IN];
static _Float16 A_f16[BATCH * IN];

static float    W_f32[IN * OUT];          // master weights (FP32)
static _Float16 W_f16[IN * OUT];          // FP16 view of the same weights

static float    bias_f32[OUT];

static float    dout_f32[BATCH * OUT];    // upstream gradient dL/dy (FP32)
static _Float16 dout_f16[BATCH * OUT];    // FP16 view (for standalone nt/tn tests)

// Generic result buffers, max element count = IN*OUT (the weight-gradient shape).
static float    R16[IN * OUT];            // result produced by the FP16 path
static float    R32[IN * OUT];            // result produced by the FP32 path

// fc_op forward/backward output buffers.
static float    out16[BATCH * OUT];
static float    out32[BATCH * OUT];
static float    dinput16[BATCH * IN];
static float    dinput32[BATCH * IN];
static float    dweights16[IN * OUT];
static float    dweights32[IN * OUT];
static float    dbias16[OUT];
static float    dbias32[OUT];

// Optimizer / flow velocity (FP32 master) buffers.
static float    v_w32[IN * OUT];
static float    v_w16[IN * OUT];          // velocity stays FP32 even for FP16 weights

// Pure FP16 buffers — output of _16-suffixed ops is _Float16, not float.
static _Float16 bias_f16[OUT];            // FP16 bias for matrix_multiply_fused_16 / fc_op_forward_p16
static _Float16 Rp16[IN * OUT];           // raw pure FP16 matrix result
static float    Rp16_w[IN * OUT];         // widened to float for compare_f32
static _Float16 outp16[BATCH * OUT];      // pure FP16 forward output
static float    outp16_w[BATCH * OUT];    // widened

// ===========================================================================
//  Deterministic data generation (so FP16 and FP32 paths get identical inputs).
// ===========================================================================
static uint32_t g_rng = 0x1234567u;
static float frand_sym(float scale)
{
    g_rng = g_rng * 1664525u + 1013904223u;
    float u = ((g_rng >> 8) & 0xFFFFFF) / (float)0x1000000;  // [0,1)
    return (u * 2.0f - 1.0f) * scale;                        // [-scale, scale)
}

static void widen_f16_to_f32(const _Float16 *src, float *dst, int n)
{
    for (int i = 0; i < n; i++) dst[i] = (float)src[i];
}

static void cast_f32_to_f16(const float *src, _Float16 *dst, int n)
{
    for (int i = 0; i < n; i++) dst[i] = (_Float16)src[i];
}

// ===========================================================================
//  Comparison + reporting.
// ===========================================================================
typedef struct {
    double max_abs;
    double max_rel;
    double mean_abs;
    int    arg_max;   // index of worst absolute error
} errstat;

static errstat compare_f32(const float *a, const float *b, int n)
{
    errstat e = {0.0, 0.0, 0.0, 0};
    double sum_abs = 0.0;
    for (int i = 0; i < n; i++) {
        double da = (double)a[i] - (double)b[i];
        double ad = da < 0 ? -da : da;
        double ref = (double)b[i];
        double aref = ref < 0 ? -ref : ref;
        double rel = ad / (aref > ABS_FLOOR ? aref : ABS_FLOOR);
        if (ad > e.max_abs) { e.max_abs = ad; e.arg_max = i; }
        if (rel > e.max_rel) e.max_rel = rel;
        sum_abs += ad;
    }
    e.mean_abs = sum_abs / (n > 0 ? n : 1);
    return e;
}

static int g_pass = 0, g_total = 0;

static void report(const char *name, errstat e, int n)
{
    g_total++;
    int ok = (e.max_rel <= REL_TOL);
    if (ok) g_pass++;
    // %f-only formatting for the bundled printf_.
    printf_("  [%s] %-26s  n=%-8d  max_abs=%.8f  max_rel=%.6f  mean_abs=%.8f  (worst idx %d)\n",
            ok ? "PASS" : "CHECK", name, n,
            (double)e.max_abs, (double)e.max_rel, (double)e.mean_abs, e.arg_max);
}

// ===========================================================================
//  Copied loss + optimizer functions (verbatim from the two train.c files,
//  renamed). Per project decision these are COPIED into the harness rather
//  than linked (they are `static` inside train.c). They run the real RVV asm.
// ===========================================================================
static inline void CLIP(float *x, float down, float up) { *x = MIN(up, MAX(down, *x)); }

// --- MSE backward (identical float math in both trees; operates on FP32 preds) ---
static float harness_mse_loss_vec(float *delta_preds, const float *preds,
                                  const float *targets, int units, int BATCH_SIZE)
{
    int total_elems = BATCH_SIZE * units;
    float scale_factor = 0.5f / (float)total_elems;
    float grad_scale = 1.0f / (float)total_elems;

    size_t max_vl;
    asm volatile("vsetvli %0, zero, e32, m8, ta, ma" : "=r"(max_vl));
    asm volatile("vmv.v.i v8, 0");

    int n = total_elems;
    const float *p_ptr = preds;
    const float *t_ptr = targets;
    float *d_ptr = delta_preds;

    while (n > 0) {
        size_t vl;
        asm volatile("vsetvli %0, %1, e32, m8, ta, ma" : "=r"(vl) : "r"(n));
        asm volatile("vle32.v v16, (%0)" :: "r"(p_ptr));
        asm volatile("vle32.v v24, (%0)" :: "r"(t_ptr));
        asm volatile("vfsub.vv v16, v16, v24");
        asm volatile("vfmacc.vv v8, v16, v16");
        asm volatile("vfmul.vf v16, v16, %0" :: "f"(grad_scale));
        asm volatile("vse32.v v16, (%0)" :: "r"(d_ptr));
        p_ptr += vl; t_ptr += vl; d_ptr += vl; n -= vl;
    }

    asm volatile("vsetvli zero, %0, e32, m8, ta, ma" :: "r"(max_vl));
    float zero = 0.0f;
    asm volatile("vfmv.s.f v0, %0" :: "f"(zero));
    asm volatile("vfredusum.vs v0, v8, v0");
    float sum_squares;
    asm volatile("vfmv.f.s %0, v0" : "=f"(sum_squares));
    return sum_squares * scale_factor;
}

// --- Cross-entropy backward (scalar, identical float math in both trees) ---
static float harness_cross_entropy_loss(float *delta_preds, const float *preds,
                                        const int *labels, int units, int BATCH_SIZE)
{
    float ce_loss = 0;
    for (int p = 0; p < BATCH_SIZE; p++) {
        float max_val = preds[p * units];
        for (int i = 1; i < units; i++)
            if (preds[i + p * units] > max_val) max_val = preds[i + p * units];
        float esum = 0;
        for (int i = 0; i < units; i++)
            esum += exp(preds[i + p * units] - max_val);
        ce_loss += 0 - log(exp(preds[labels[p] + p * units] - max_val) / esum);
        for (int i = 0; i < units; i++) {
            if (labels[p] == i)
                delta_preds[p * units + i] = exp(preds[i + p * units] - max_val) / esum - 1;
            else
                delta_preds[p * units + i] = exp(preds[i + p * units] - max_val) / esum;
        }
    }
    return ce_loss / BATCH_SIZE;
}

// --- FP32 momentum SGD (used for biases in both trees, and FP32 reference) ---
static void harness_momentum_sgd_vec_f32(float *w, float *v_w, const float *d_w, int units)
{
    float lr = LEARNING_RATE;
    int n = units;
#if ALEXNET_USE_MOMENTUM
    float momentum = 0.9f, clip_min = -1.0f, clip_max = 1.0f;
    while (n > 0) {
        size_t vl;
        asm volatile("vsetvli %0, %1, e32, m8, ta, ma" : "=r"(vl) : "r"(n));
        asm volatile("vle32.v v8,  (%0)" :: "r"(v_w));
        asm volatile("vle32.v v16, (%0)" :: "r"(d_w));
        asm volatile("vle32.v v24, (%0)" :: "r"(w));
        asm volatile("vfmul.vf v8, v8, %0" :: "f"(momentum));
        asm volatile("vfnmsac.vf v8, %0, v16" :: "f"(lr));
        asm volatile("vfmax.vf v8, v8, %0" :: "f"(clip_min));
        asm volatile("vfmin.vf v8, v8, %0" :: "f"(clip_max));
        asm volatile("vfadd.vv v24, v24, v8");
        asm volatile("vse32.v v8,  (%0)" :: "r"(v_w));
        asm volatile("vse32.v v24, (%0)" :: "r"(w));
        w += vl; v_w += vl; d_w += vl; n -= vl;
    }
#else
    (void)v_w;
    while (n > 0) {
        size_t vl;
        asm volatile("vsetvli %0, %1, e32, m8, ta, ma" : "=r"(vl) : "r"(n));
        asm volatile("vle32.v v8,  (%0)" :: "r"(w));
        asm volatile("vle32.v v16, (%0)" :: "r"(d_w));
        asm volatile("vfnmsac.vf v8, %0, v16" :: "f"(lr));
        asm volatile("vse32.v v8,  (%0)" :: "r"(w));
        w += vl; d_w += vl; n -= vl;
    }
#endif
}

// --- FP16-weight momentum SGD: widen FP16->FP32, do FP32 math, narrow back ---
static void harness_momentum_sgd_vec_f16(_Float16 *w, float *v_w, const float *d_w, int units)
{
    float lr = LEARNING_RATE;
    int n = units;
#if ALEXNET_USE_MOMENTUM
    float momentum = 0.9f, clip_min = -1.0f, clip_max = 1.0f;
    while (n > 0) {
        size_t vl;
        asm volatile("vsetvli %0, %1, e16, m4, ta, ma" : "=r"(vl) : "r"(n));
        asm volatile("vle16.v v20, (%0)" :: "r"(w));
        asm volatile("vfwcvt.f.f.v v24, v20");
        asm volatile("vsetvli zero, %0, e32, m8, ta, ma" :: "r"(vl));
        asm volatile("vle32.v v8,  (%0)" :: "r"(v_w));
        asm volatile("vle32.v v16, (%0)" :: "r"(d_w));
        asm volatile("vfmul.vf  v8, v8, %0"   :: "f"(momentum));
        asm volatile("vfnmsac.vf v8, %0, v16" :: "f"(lr));
        asm volatile("vfmax.vf  v8, v8, %0"   :: "f"(clip_min));
        asm volatile("vfmin.vf  v8, v8, %0"   :: "f"(clip_max));
        asm volatile("vfadd.vv  v24, v24, v8");
        asm volatile("vse32.v   v8,  (%0)" :: "r"(v_w));
        asm volatile("vsetvli zero, %0, e16, m4, ta, ma" :: "r"(vl));
        asm volatile("vfncvt.f.f.w v20, v24");
        asm volatile("vse16.v v20, (%0)" :: "r"(w));
        w += vl; v_w += vl; d_w += vl; n -= vl;
    }
#else
    (void)v_w;
    while (n > 0) {
        size_t vl;
        asm volatile("vsetvli %0, %1, e16, m4, ta, ma" : "=r"(vl) : "r"(n));
        asm volatile("vle16.v v20, (%0)" :: "r"(w));
        asm volatile("vfwcvt.f.f.v v24, v20");
        asm volatile("vsetvli zero, %0, e32, m8, ta, ma" :: "r"(vl));
        asm volatile("vle32.v v16, (%0)" :: "r"(d_w));
        asm volatile("vfnmsac.vf v24, %0, v16" :: "f"(lr));
        asm volatile("vsetvli zero, %0, e16, m4, ta, ma" :: "r"(vl));
        asm volatile("vfncvt.f.f.w v20, v24");
        asm volatile("vse16.v v20, (%0)" :: "r"(w));
        w += vl; d_w += vl; n -= vl;
    }
#endif
}

// ===========================================================================
//  Shared input initialization.
// ===========================================================================
static void init_inputs(void)
{
    // Activations straight from data.S (first BATCH samples).
    for (int i = 0; i < BATCH * IN; i++) A_f32[i] = test_inputs[i];
    cast_f32_to_f16(A_f32, A_f16, BATCH * IN);

    // Deterministic small weights (so both paths are identical before rounding).
    for (int i = 0; i < IN * OUT; i++) W_f32[i] = frand_sym(0.1f);
    cast_f32_to_f16(W_f32, W_f16, IN * OUT);

    // Bias from a deterministic sequence.
    for (int i = 0; i < OUT; i++) bias_f32[i] = frand_sym(0.05f);

    // A plausible upstream gradient dL/dy.
    for (int i = 0; i < BATCH * OUT; i++) dout_f32[i] = frand_sym(0.5f);
    cast_f32_to_f16(dout_f32, dout_f16, BATCH * OUT);

    // FP16 bias for pure-FP16 ops (fc_op_p16.bias is _Float16 *).
    cast_f32_to_f16(bias_f32, bias_f16, OUT);
}

// ===========================================================================
//  PART 1 - one-by-one kernel comparisons.
// ===========================================================================
static void test_matmul_variants(void)
{
    printf_("\n--- PART 1a: matrix_multiply three-way (mixed FP16 | pure FP16 | FP32) ---\n");

    // Each kernel: compute all three results once, then report all three pairs.
    // [mixed/fp32]  [pure/fp32]  [mixed/pure]

    // C[BATCH x OUT] = A[BATCH x IN] . W[IN x OUT]
    memset(R16, 0, sizeof(float) * BATCH * OUT);
    memset(R32, 0, sizeof(float) * BATCH * OUT);
    memset(Rp16, 0, sizeof(Rp16));
    matrix_multiply(A_f16, W_f16, R16, BATCH, IN, OUT);
    matrix_multiply_32(A_f32, W_f32, R32, BATCH, IN, OUT);
    matrix_multiply_16(A_f16, W_f16, Rp16, BATCH, IN, OUT);
    widen_f16_to_f32(Rp16, Rp16_w, BATCH * OUT);
    report("matrix_multiply     [mixed/fp32]", compare_f32(R16,    R32,    BATCH * OUT), BATCH * OUT);
    report("matrix_multiply_16  [pure/fp32] ", compare_f32(Rp16_w, R32,    BATCH * OUT), BATCH * OUT);
    report("matrix_multiply     [mixed/pure]", compare_f32(R16,    Rp16_w, BATCH * OUT), BATCH * OUT);

    // Fused: C = A.W + bias  (bias is float for mixed/fp32, _Float16 for pure)
    memset(R16, 0, sizeof(float) * BATCH * OUT);
    memset(R32, 0, sizeof(float) * BATCH * OUT);
    memset(Rp16, 0, sizeof(Rp16));
    matrix_multiply_fused(A_f16, W_f16, bias_f32, R16, BATCH, IN, OUT);
    matrix_multiply_fused_32(A_f32, W_f32, bias_f32, R32, BATCH, IN, OUT);
    matrix_multiply_fused_16(A_f16, W_f16, bias_f16, Rp16, BATCH, IN, OUT);
    widen_f16_to_f32(Rp16, Rp16_w, BATCH * OUT);
    report("matrix_multiply_fused     [mixed/fp32]", compare_f32(R16,    R32,    BATCH * OUT), BATCH * OUT);
    report("matrix_multiply_fused_16  [pure/fp32] ", compare_f32(Rp16_w, R32,    BATCH * OUT), BATCH * OUT);
    report("matrix_multiply_fused     [mixed/pure]", compare_f32(R16,    Rp16_w, BATCH * OUT), BATCH * OUT);

    // NT (d_input shape): C[BATCH x IN] = dout[BATCH x OUT] . W[IN x OUT]^T
    memset(R16, 0, sizeof(float) * BATCH * IN);
    memset(R32, 0, sizeof(float) * BATCH * IN);
    memset(Rp16, 0, sizeof(Rp16));
    matrix_multiply_nt(dout_f16, W_f16, R16, BATCH, OUT, IN);
    matrix_multiply_nt_32(dout_f32, W_f32, R32, BATCH, OUT, IN);
    matrix_multiply_nt_16(dout_f16, W_f16, Rp16, BATCH, OUT, IN);
    widen_f16_to_f32(Rp16, Rp16_w, BATCH * IN);
    report("matrix_multiply_nt     [mixed/fp32]", compare_f32(R16,    R32,    BATCH * IN), BATCH * IN);
    report("matrix_multiply_nt_16  [pure/fp32] ", compare_f32(Rp16_w, R32,    BATCH * IN), BATCH * IN);
    report("matrix_multiply_nt     [mixed/pure]", compare_f32(R16,    Rp16_w, BATCH * IN), BATCH * IN);

    // NT deferred (same shape as NT)
    memset(R16, 0, sizeof(float) * BATCH * IN);
    memset(R32, 0, sizeof(float) * BATCH * IN);
    memset(Rp16, 0, sizeof(Rp16));
    matrix_multiply_nt_deferred(dout_f16, W_f16, R16, BATCH, OUT, IN);
    matrix_multiply_nt_deferred_32(dout_f32, W_f32, R32, BATCH, OUT, IN);
    matrix_multiply_nt_deferred_16(dout_f16, W_f16, Rp16, BATCH, OUT, IN);
    widen_f16_to_f32(Rp16, Rp16_w, BATCH * IN);
    report("matrix_multiply_nt_deferred     [mixed/fp32]", compare_f32(R16,    R32,    BATCH * IN), BATCH * IN);
    report("matrix_multiply_nt_deferred_16  [pure/fp32] ", compare_f32(Rp16_w, R32,    BATCH * IN), BATCH * IN);
    report("matrix_multiply_nt_deferred     [mixed/pure]", compare_f32(R16,    Rp16_w, BATCH * IN), BATCH * IN);

    // TN (d_weights shape): C[IN x OUT] = A[BATCH x IN]^T . dout[BATCH x OUT]
    memset(R16, 0, sizeof(float) * IN * OUT);
    memset(R32, 0, sizeof(float) * IN * OUT);
    memset(Rp16, 0, sizeof(Rp16));
    matrix_multiply_tn(A_f16, dout_f16, R16, BATCH, IN, OUT);
    matrix_multiply_tn_32(A_f32, dout_f32, R32, BATCH, IN, OUT);
    matrix_multiply_tn_16(A_f16, dout_f16, Rp16, BATCH, IN, OUT);
    widen_f16_to_f32(Rp16, Rp16_w, IN * OUT);
    report("matrix_multiply_tn     [mixed/fp32]", compare_f32(R16,    R32,    IN * OUT), IN * OUT);
    report("matrix_multiply_tn_16  [pure/fp32] ", compare_f32(Rp16_w, R32,    IN * OUT), IN * OUT);
    report("matrix_multiply_tn     [mixed/pure]", compare_f32(R16,    Rp16_w, IN * OUT), IN * OUT);
}

static void test_forward(void)
{
    printf_("\n--- PART 1b: fc_op_forward (full layer) ---\n");

    fc_op op16;
    memset(&op16, 0, sizeof(op16));
    op16.input = A_f16; op16.weights = W_f16; op16.bias = bias_f32; op16.output = out16;
    op16.in_units = IN; op16.out_units = OUT; op16.batchsize = BATCH; op16.layer_id = 1;
    memset(out16, 0, sizeof(out16));
    fc_op_forward(&op16);

    fc_op_32 op32;
    memset(&op32, 0, sizeof(op32));
    op32.input = A_f32; op32.weights = W_f32; op32.bias = bias_f32; op32.output = out32;
    op32.in_units = IN; op32.out_units = OUT; op32.batchsize = BATCH; op32.layer_id = 1;
    memset(out32, 0, sizeof(out32));
    fc_op_forward_32(&op32);

    // Pure FP16 forward: all-FP16 struct, _Float16 * output.
    fc_op_p16 opp16;
    memset(&opp16, 0, sizeof(opp16));
    opp16.input    = A_f16;
    opp16.weights  = W_f16;
    opp16.bias     = bias_f16;
    opp16.output   = outp16;
    opp16.in_units = IN; opp16.out_units = OUT; opp16.batchsize = BATCH; opp16.layer_id = 1;
    memset(outp16, 0, sizeof(outp16));
    fc_op_forward_p16(&opp16);
    widen_f16_to_f32(outp16, outp16_w, BATCH * OUT);

    report("fc_op_forward  [mixed/fp32]", compare_f32(out16,    out32,    BATCH * OUT), BATCH * OUT);
    report("fc_op_forward  [pure/fp32] ", compare_f32(outp16_w, out32,    BATCH * OUT), BATCH * OUT);
    report("fc_op_forward  [mixed/pure]", compare_f32(out16,    outp16_w, BATCH * OUT), BATCH * OUT);
}

static void test_backward(void)
{
    printf_("\n--- PART 1c: fc_op_backward_full_profile ---\n");

    // Outputs are accumulated into (M padded > M), so pre-zero d_input / d_weights.
    memset(dinput16, 0, sizeof(dinput16)); memset(dweights16, 0, sizeof(dweights16)); memset(dbias16, 0, sizeof(dbias16));
    memset(dinput32, 0, sizeof(dinput32)); memset(dweights32, 0, sizeof(dweights32)); memset(dbias32, 0, sizeof(dbias32));

    fc_op op16;
    memset(&op16, 0, sizeof(op16));
    op16.input = A_f16; op16.weights = W_f16; op16.bias = bias_f32;
    op16.d_output = dout_f32;     // FP32 upstream gradient (layer downcasts internally)
    op16.d_input = dinput16; op16.d_weights = dweights16; op16.d_bias = dbias16;
    op16.in_units = IN; op16.out_units = OUT; op16.batchsize = BATCH; op16.layer_id = 1;
    fc_backward_cycle_breakdown cyc16 = {0, 0, 0};
    fc_op_backward_full_profile(&op16, &cyc16);

    fc_op_32 op32;
    memset(&op32, 0, sizeof(op32));
    op32.input = A_f32; op32.weights = W_f32; op32.bias = bias_f32;
    op32.d_output = dout_f32;
    op32.d_input = dinput32; op32.d_weights = dweights32; op32.d_bias = dbias32;
    op32.in_units = IN; op32.out_units = OUT; op32.batchsize = BATCH; op32.layer_id = 1;
    fc_bw_cyc_32 cyc32 = {0, 0, 0};
    fc_op_backward_full_profile_32(&op32, &cyc32);

    report("backward.d_input",   compare_f32(dinput16,   dinput32,   BATCH * IN), BATCH * IN);
    report("backward.d_weights", compare_f32(dweights16, dweights32, IN * OUT),   IN * OUT);
    report("backward.d_bias",    compare_f32(dbias16,    dbias32,    OUT),        OUT);
}

static void test_loss(void)
{
    printf_("\n--- PART 1d: loss functions (on each path's forward output) ---\n");
    // out16 / out32 / outp16_w were produced in test_forward().
    static float grad16[BATCH * OUT];
    static float grad32[BATCH * OUT];
    static float gradp16[BATCH * OUT];
    static float tgt[BATCH * OUT];
    for (int i = 0; i < BATCH * OUT; i++) tgt[i] = test_targets[i];

    float mse16  = harness_mse_loss_vec(grad16,   out16,    tgt, OUT, BATCH);
    float mse32  = harness_mse_loss_vec(grad32,   out32,    tgt, OUT, BATCH);
    float msep16 = harness_mse_loss_vec(gradp16,  outp16_w, tgt, OUT, BATCH);
    printf_("  MSE: mixed=%.8f  pure=%.8f  fp32=%.8f\n",
            (double)mse16, (double)msep16, (double)mse32);
    report("mse.grad  [mixed/fp32]", compare_f32(grad16,  grad32,   BATCH * OUT), BATCH * OUT);
    report("mse.grad  [pure/fp32] ", compare_f32(gradp16, grad32,   BATCH * OUT), BATCH * OUT);
    report("mse.grad  [mixed/pure]", compare_f32(grad16,  gradp16,  BATCH * OUT), BATCH * OUT);

    int lbl[BATCH];
    for (int i = 0; i < BATCH; i++) lbl[i] = test_labels[i];
    float ce16  = harness_cross_entropy_loss(grad16,  out16,    lbl, OUT, BATCH);
    float ce32  = harness_cross_entropy_loss(grad32,  out32,    lbl, OUT, BATCH);
    float cep16 = harness_cross_entropy_loss(gradp16, outp16_w, lbl, OUT, BATCH);
    printf_("  CE:  mixed=%.8f  pure=%.8f  fp32=%.8f\n",
            (double)ce16, (double)cep16, (double)ce32);
    report("ce.grad  [mixed/fp32]", compare_f32(grad16,  grad32,  BATCH * OUT), BATCH * OUT);
    report("ce.grad  [pure/fp32] ", compare_f32(gradp16, grad32,  BATCH * OUT), BATCH * OUT);
    report("ce.grad  [mixed/pure]", compare_f32(grad16,  gradp16, BATCH * OUT), BATCH * OUT);
}

static void test_optimizer(void)
{
    printf_("\n--- PART 1e: momentum SGD update (FP16-weight vs FP32-weight) ---\n");
    const int N = 8192;   // several vector iterations; same logic as full IN*OUT
    static float    w32[8192], vel32[8192], grad[8192];
    static _Float16 w16[8192];
    static float    vel16[8192];
    static float    w16_widened[8192];

    for (int i = 0; i < N; i++) {
        w32[i]  = frand_sym(0.1f);
        w16[i]  = (_Float16)w32[i];
        vel32[i] = 0.0f; vel16[i] = 0.0f;
        grad[i]  = frand_sym(1.0f);
    }

    harness_momentum_sgd_vec_f32(w32, vel32, grad, N);
    harness_momentum_sgd_vec_f16(w16, vel16, grad, N);
    widen_f16_to_f32(w16, w16_widened, N);

    report("momentum_sgd.weights", compare_f32(w16_widened, w32, N), N);
    report("momentum_sgd.velocity", compare_f32(vel16, vel32, N), N);
}

// ===========================================================================
//  PART 2 - end-to-end training step ("in flow").
// ===========================================================================
static void test_flow(int steps)
{
    printf_("\n--- PART 2: end-to-end training step (forward->MSE->backward->update) ---\n");

    // Fresh master weights for both precisions.
    for (int i = 0; i < IN * OUT; i++) { W_f32[i] = frand_sym(0.1f); v_w32[i] = 0.0f; v_w16[i] = 0.0f; }
    cast_f32_to_f16(W_f32, W_f16, IN * OUT);

    static float tgt[BATCH * OUT];
    static float grad16[BATCH * OUT];
    static float grad32[BATCH * OUT];
    static float w16_widened[IN * OUT];
    for (int i = 0; i < BATCH * OUT; i++) tgt[i] = test_targets[i];

    for (int s = 0; s < steps; s++) {
        // ---- forward ----
        fc_op op16; memset(&op16, 0, sizeof(op16));
        op16.input = A_f16; op16.weights = W_f16; op16.bias = bias_f32; op16.output = out16;
        op16.in_units = IN; op16.out_units = OUT; op16.batchsize = BATCH; op16.layer_id = 1;
        memset(out16, 0, sizeof(out16));
        fc_op_forward(&op16);

        fc_op_32 op32; memset(&op32, 0, sizeof(op32));
        op32.input = A_f32; op32.weights = W_f32; op32.bias = bias_f32; op32.output = out32;
        op32.in_units = IN; op32.out_units = OUT; op32.batchsize = BATCH; op32.layer_id = 1;
        memset(out32, 0, sizeof(out32));
        fc_op_forward_32(&op32);

        // ---- loss / upstream gradient ----
        float loss16 = harness_mse_loss_vec(grad16, out16, tgt, OUT, BATCH);
        float loss32 = harness_mse_loss_vec(grad32, out32, tgt, OUT, BATCH);

        // ---- backward ----
        memset(dinput16, 0, sizeof(dinput16)); memset(dweights16, 0, sizeof(dweights16)); memset(dbias16, 0, sizeof(dbias16));
        memset(dinput32, 0, sizeof(dinput32)); memset(dweights32, 0, sizeof(dweights32)); memset(dbias32, 0, sizeof(dbias32));

        op16.d_output = grad16; op16.d_input = dinput16; op16.d_weights = dweights16; op16.d_bias = dbias16;
        fc_backward_cycle_breakdown cyc16 = {0, 0, 0};
        fc_op_backward_full_profile(&op16, &cyc16);

        op32.d_output = grad32; op32.d_input = dinput32; op32.d_weights = dweights32; op32.d_bias = dbias32;
        fc_bw_cyc_32 cyc32 = {0, 0, 0};
        fc_op_backward_full_profile_32(&op32, &cyc32);

        // ---- weight update ----
        harness_momentum_sgd_vec_f16(W_f16, v_w16, dweights16, IN * OUT);
        harness_momentum_sgd_vec_f32(W_f32, v_w32, dweights32, IN * OUT);

        widen_f16_to_f32(W_f16, w16_widened, IN * OUT);
        errstat ew = compare_f32(w16_widened, W_f32, IN * OUT);
        printf_("  step %d:  loss16=%.8f  loss32=%.8f  |dloss|=%.8f  weight max_rel=%.6f  weight max_abs=%.8f\n",
                s, (double)loss16, (double)loss32,
                (double)fabs((double)loss16 - (double)loss32),
                (double)ew.max_rel, (double)ew.max_abs);
    }
    // Final weight divergence as a pass/fail line.
    errstat ew = compare_f32(w16_widened, W_f32, IN * OUT);
    report("flow.final_weights", ew, IN * OUT);
}

// ===========================================================================
int main(void)
{
    printf_("=====================================================================\n");
    printf_(" FC-layer precision comparison: mixed FP16 | pure FP16 | FP32\n");
    printf_("   IN=%d  OUT=%d  BATCH=%d   REL_TOL=%.4f\n", IN, OUT, BATCH, (double)REL_TOL);
    printf_("=====================================================================\n");

    init_inputs();

    test_matmul_variants();
    test_forward();
    test_backward();
    test_loss();
    test_optimizer();

    test_flow(3);

    printf_("\n=====================================================================\n");
    printf_(" SUMMARY: %d/%d comparisons within REL_TOL=%.4f\n", g_pass, g_total, (double)REL_TOL);
    printf_("   (CHECK lines are not necessarily bugs: FP16 rounding can legitimately\n");
    printf_("    exceed the tolerance on ill-conditioned reductions - inspect max_rel.)\n");
    printf_("=====================================================================\n");
    return 0;
}
