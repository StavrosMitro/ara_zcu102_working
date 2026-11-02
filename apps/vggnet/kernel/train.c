//
// File:        train.c
// Description: FP32 training for MiniVGGNet.
// MiniVGGNet rewrite + vectorization: Stavros Mitropoulos, NTUA
//
// The backward pass covers the whole network:
//
//   fc2 -> bn5 -> relu5 -> fc1 -> pool2
//       -> bn4 -> relu4 -> conv4 -> bn3 -> relu3 -> conv3 -> pool1
//       -> bn2 -> relu2 -> conv2 -> bn1 -> relu1 -> conv1
//
// Each layer's trainable flag selects *_backward_full (weights + input) or
// *_backward_input_only (input only). When the entire conv stack is frozen
// (FINETUNE_MODE) the chain stops at fc1, since nothing below it would use
// the gradient.
//
// BN normalises with batch statistics on every forward pass, so a short
// trailing batch would skew them; training drops it.
//
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
#include "alexnet.h"
#include "data.h"
#ifdef SPIKE
#include <printf.h>
#elif defined ARA_LINUX
#include <stdio.h>
#else
#include "printf.h"
#endif

#include "utils.h"
#include "activation_layer.h"

#define MAX(a,b) (((a) > (b)) ? (a) : (b))
#define MIN(a,b) (((a) < (b)) ? (a) : (b))

#if defined(ALEXNET_LAYER_LOGS) && !defined(SPIKE)
#define ALEXNET_LOG_LAYER(...) printf_(__VA_ARGS__)
#else
#define ALEXNET_LOG_LAYER(...)
#endif

#ifndef ALEXNET_STATIC_MAX_BATCH
#ifdef ALEXNET_BATCHSIZE
#define ALEXNET_STATIC_MAX_BATCH ALEXNET_BATCHSIZE
#else
#define ALEXNET_STATIC_MAX_BATCH 1
#endif
#endif

#ifndef ALEXNET_MAX_STEPS
#define ALEXNET_MAX_STEPS 0
#endif

#ifndef LEARNING_RATE
#ifdef FINETUNE_MODE
#define LEARNING_RATE 1e-3f      // head starts from pre-trained weights
#else
#define LEARNING_RATE 0.01f      // training from scratch
#endif
#endif

#ifndef MOMENTUM
#define MOMENTUM 0.9f
#endif

#ifndef WEIGHT_DECAY
#define WEIGHT_DECAY 5e-4f
#endif

#ifndef LR_STEP_EPOCHS
#define LR_STEP_EPOCHS 20
#endif

#ifndef LR_STEP_GAMMA
#define LR_STEP_GAMMA 0.1f
#endif

// Percentage of the dataset held out for evaluation and never trained on.
#ifndef EVAL_PERCENT
#define EVAL_PERCENT 20
#endif

#ifndef RANDOM_SEED
#define RANDOM_SEED 1
#endif

// Training-time data augmentation (random horizontal flip). 1 = on, 0 = off.
#ifndef AUGMENT
#define AUGMENT 1
#endif

// Dropout masks, produced by the forward pass in main.c (same batch).
extern float do1_mask[];    // after pool1
extern float do2_mask[];    // after pool2
extern float dofc_mask[];   // after bn5

#define CONV_KK      (CONV_KERNEL_L * CONV_KERNEL_L)
#define IMAGE_UNITS  (IN_CHANNELS * FEATURE0_L * FEATURE0_L)
#define BLOCK1_UNITS (C1_CHANNELS * FEATURE1_L * FEATURE1_L)
#define POOL1_UNITS  (C1_CHANNELS * POOLING1_L * POOLING1_L)
#define BLOCK2_UNITS (C2_CHANNELS * FEATURE2_L * FEATURE2_L)
#define POOL2_UNITS  (C2_CHANNELS * POOLING2_L * POOLING2_L)

#define CONV1_W (C1_CHANNELS * IN_CHANNELS * CONV_KK)
#define CONV2_W (C1_CHANNELS * C1_CHANNELS * CONV_KK)
#define CONV3_W (C2_CHANNELS * C1_CHANNELS * CONV_KK)
#define CONV4_W (C2_CHANNELS * C2_CHANNELS * CONV_KK)
#define FC1_W   (FC1_IN_UNITS * FC1_LAYER)
#define FC2_W   (FC1_LAYER * OUT_LAYER)

// --------------------------------------------------------------------------
// Weight-gradient buffers
// --------------------------------------------------------------------------

static float d_conv1_weights[CONV1_W]; static float d_conv1_bias[C1_CHANNELS];
static float d_conv2_weights[CONV2_W]; static float d_conv2_bias[C1_CHANNELS];
static float d_conv3_weights[CONV3_W]; static float d_conv3_bias[C2_CHANNELS];
static float d_conv4_weights[CONV4_W]; static float d_conv4_bias[C2_CHANNELS];
static float d_fc1_weights[FC1_W];     static float d_fc1_bias[FC1_LAYER];
static float d_fc2_weights[FC2_W];     static float d_fc2_bias[OUT_LAYER];

static float d_bn1_gamma[C1_CHANNELS]; static float d_bn1_beta[C1_CHANNELS];
static float d_bn2_gamma[C1_CHANNELS]; static float d_bn2_beta[C1_CHANNELS];
static float d_bn3_gamma[C2_CHANNELS]; static float d_bn3_beta[C2_CHANNELS];
static float d_bn4_gamma[C2_CHANNELS]; static float d_bn4_beta[C2_CHANNELS];
static float d_bn5_gamma[FC1_LAYER];   static float d_bn5_beta[FC1_LAYER];

// --------------------------------------------------------------------------
// Momentum velocities, one per trainable parameter tensor
// --------------------------------------------------------------------------

static float v_conv1_weights[CONV1_W]; static float v_conv1_bias[C1_CHANNELS];
static float v_conv2_weights[CONV2_W]; static float v_conv2_bias[C1_CHANNELS];
static float v_conv3_weights[CONV3_W]; static float v_conv3_bias[C2_CHANNELS];
static float v_conv4_weights[CONV4_W]; static float v_conv4_bias[C2_CHANNELS];
static float v_fc1_weights[FC1_W];     static float v_fc1_bias[FC1_LAYER];
static float v_fc2_weights[FC2_W];     static float v_fc2_bias[OUT_LAYER];

static float v_bn1_gamma[C1_CHANNELS]; static float v_bn1_beta[C1_CHANNELS];
static float v_bn2_gamma[C1_CHANNELS]; static float v_bn2_beta[C1_CHANNELS];
static float v_bn3_gamma[C2_CHANNELS]; static float v_bn3_beta[C2_CHANNELS];
static float v_bn4_gamma[C2_CHANNELS]; static float v_bn4_beta[C2_CHANNELS];
static float v_bn5_gamma[FC1_LAYER];   static float v_bn5_beta[FC1_LAYER];

// --------------------------------------------------------------------------
// Activation gradients along the backward chain.
// d_<layer>_out holds the gradient w.r.t. that layer's OUTPUT.
// --------------------------------------------------------------------------

static float d_fc2_out  [ALEXNET_STATIC_MAX_BATCH * OUT_LAYER];   // loss grad at the logits
static float d_bn5_out  [ALEXNET_STATIC_MAX_BATCH * FC1_LAYER];
static float d_relu5_out[ALEXNET_STATIC_MAX_BATCH * FC1_LAYER];
static float d_fc1_out  [ALEXNET_STATIC_MAX_BATCH * FC1_LAYER];
static float d_pool2_out[ALEXNET_STATIC_MAX_BATCH * POOL2_UNITS];

static float d_bn4_out  [ALEXNET_STATIC_MAX_BATCH * BLOCK2_UNITS];
static float d_relu4_out[ALEXNET_STATIC_MAX_BATCH * BLOCK2_UNITS];
static float d_conv4_out[ALEXNET_STATIC_MAX_BATCH * BLOCK2_UNITS];
static float d_bn3_out  [ALEXNET_STATIC_MAX_BATCH * BLOCK2_UNITS];
static float d_relu3_out[ALEXNET_STATIC_MAX_BATCH * BLOCK2_UNITS];
static float d_conv3_out[ALEXNET_STATIC_MAX_BATCH * BLOCK2_UNITS];
static float d_pool1_out[ALEXNET_STATIC_MAX_BATCH * POOL1_UNITS];

static float d_bn2_out  [ALEXNET_STATIC_MAX_BATCH * BLOCK1_UNITS];
static float d_relu2_out[ALEXNET_STATIC_MAX_BATCH * BLOCK1_UNITS];
static float d_conv2_out[ALEXNET_STATIC_MAX_BATCH * BLOCK1_UNITS];
static float d_bn1_out  [ALEXNET_STATIC_MAX_BATCH * BLOCK1_UNITS];
static float d_relu1_out[ALEXNET_STATIC_MAX_BATCH * BLOCK1_UNITS];
static float d_conv1_out[ALEXNET_STATIC_MAX_BATCH * BLOCK1_UNITS];

// conv1's d_input: the gradient w.r.t. the image. Computed, then discarded.
static float d_image[ALEXNET_STATIC_MAX_BATCH * IMAGE_UNITS];

static float train_input_buf  [ALEXNET_STATIC_MAX_BATCH * IMAGE_UNITS];
static int   train_batch_Y_buf[ALEXNET_STATIC_MAX_BATCH];
static int   train_preds_buf  [ALEXNET_STATIC_MAX_BATCH];

static float test_input_buf   [ALEXNET_STATIC_MAX_BATCH * IMAGE_UNITS];
static int   test_batch_Y_buf [ALEXNET_STATIC_MAX_BATCH];
static int   test_preds_buf   [ALEXNET_STATIC_MAX_BATCH];

// Confusion counts, accumulated across a whole eval sweep
static int metrics_true_pos [OUT_LAYER];
static int metrics_false_pos[OUT_LAYER];
static int metrics_false_neg[OUT_LAYER];

// Current learning rate (stepped down per LR_STEP_EPOCHS)
static float g_lr = LEARNING_RATE;

static void zero_f32(float *buf, int n)
{
    // Routed through the vector helper: the backward pass zeroes the whole
    // activation-gradient chain each step (d_conv1_out alone is
    // batch*C1*32*32 floats), so this is a hot path, not bookkeeping.
    zero_f32_vec(buf, n);
}

// --------------------------------------------------------------------------
// Train/eval split — initialised once, before the baseline test pass
// --------------------------------------------------------------------------

static int split_ready = 0;

static void ensure_split(void)
{
    if (split_ready) return;

    srand(RANDOM_SEED);
    int total = get_dataset_count();
    dataset_split_init((total * EVAL_PERCENT) / 100);
    split_ready = 1;

    printf_("dataset split: %d train / %d eval (held out, never trained on)\n",
            get_train_count(), get_eval_count());
}

// --------------------------------------------------------------------------
// Cross-entropy loss — writes softmax-minus-onehot into delta_preds
// --------------------------------------------------------------------------

// The loss is the MEAN over the batch, so the gradient it hands back must be
// the mean too: dL/dz = (softmax - onehot) / B. Applying the 1/B here -- once,
// on B*units values -- means every downstream gradient inherits it through the
// chain rule. The layers therefore do NOT scale their own d_weights/d_bias;
// doing it per-layer costs ~100x more element-multiplies and lets a new layer
// silently forget the factor.

static float cross_entropy_loss(float *delta_preds, const float *preds,
                                const int *labels, int units, int BATCH_SIZE) {

    softmax_fc_forward_vec(preds, delta_preds, BATCH_SIZE, units);

    float total_loss = 0.0f;


    for (int p = 0; p < BATCH_SIZE; ++p) {
        

        float *grad_p = delta_preds + p * units; 
        
        int target_class = labels[p];

        float target_prob = grad_p[target_class];
        
        if (target_prob < 1e-7f) 
            target_prob = 1e-7f; 
        
        total_loss -= logf(target_prob);
        grad_p[target_class] -= 1.0f;

    }

    // The 1/B that every layer below depends on. Without it each parameter
    // gradient comes out BATCH_SIZE times too large -- see RVV_ISSUES.md
    // Issue 5, which removed the per-layer scaling on the strength of this.
    vector_scale_f32(delta_preds, 1.0f / (float)BATCH_SIZE, BATCH_SIZE * units);

    return total_loss / BATCH_SIZE;
}

// {   

//     float ce_loss = 0.0f;
//     const float inv_batch = 1.0f / (float)BATCH_SIZE;

//     for (int p = 0; p < BATCH_SIZE; p++) {
//         float max_val = preds[p * units];
//         for (int i = 1; i < units; i++)
//             if (preds[i + p * units] > max_val) max_val = preds[i + p * units];

//         float esum = 0.0f;
//         for (int i = 0; i < units; i++)
//             esum += expf(preds[i + p * units] - max_val);

//         ce_loss += -logf(expf(preds[labels[p] + p * units] - max_val) / esum);

//         for (int i = 0; i < units; i++) {
//             float softmax_i = expf(preds[i + p * units] - max_val) / esum;
//             delta_preds[p * units + i] =
//                 (softmax_i - (i == labels[p] ? 1.0f : 0.0f)) * inv_batch;
//         }
//     }

//     return ce_loss / BATCH_SIZE;
// }
// --------------------------------------------------------------------------
// Momentum SGD.  Weight decay applies to weight matrices only — not to biases
// or BN gamma/beta, where it would fight the normalisation.
// --------------------------------------------------------------------------

#ifdef DUMP_TRACE
// Defined further down with the backward-pass tracing; declared here so
// gradient_descent() can reuse it for the post-update weight checksums.
static void bw_trace(const char *name, const float *p, int n);
#endif

// Scalar reference, kept for A/B verification: build with -DSGD_SCALAR to
// select it. The vector path below must reproduce it exactly.
static void sgd_step_scalar(float *w, float *v, const float *dw, int n, float wd, float lr)
{
    for (int i = 0; i < n; i++) {
        float vi = MOMENTUM * v[i] - lr * (dw[i] + wd * w[i]);
        if (vi >  1.0f) vi =  1.0f;
        if (vi < -1.0f) vi = -1.0f;
        v[i]  = vi;
        w[i] += vi;
    }
}

// Vectorized momentum SGD, after conv_layer32/fc_layer32's momentum_sgd_vec.
// Two things that version does not have and this one needs:
//   * weight decay -- the `dw + wd*w` term, folded in with vfmacc;
//   * a runtime learning rate (g_lr follows a schedule), passed as a scalar
//     operand rather than the compile-time LEARNING_RATE constant.
// The operation order mirrors the scalar expression exactly -- t = dw + wd*w
// first, then MOMENTUM*v - lr*t -- so the two agree bit for bit rather than
// merely mathematically. wd == 0 for biases and BN gamma/beta; vfmacc by 0.0
// is exact, so those need no separate path.
static void sgd_step_vec(float *w, float *v, const float *dw, int n, float wd, float lr)
{
    const float momentum = MOMENTUM;
    const float clip_lo  = -1.0f;
    const float clip_hi  =  1.0f;

    while (n > 0) {
        size_t vl;
        asm volatile("vsetvli %0, %1, e32, m8, ta, ma" : "=r"(vl) : "r"(n));

        asm volatile("vle32.v v8,  (%0)" :: "r"(v)  : "memory");   // velocity
        asm volatile("vle32.v v16, (%0)" :: "r"(dw) : "memory");   // gradient
        asm volatile("vle32.v v24, (%0)" :: "r"(w)  : "memory");   // weights

        // v16 = dw + wd*w
        asm volatile("vfmacc.vf v16, %0, v24" :: "f"(wd));
        // v8 = MOMENTUM*v ; then v8 -= lr*v16
        asm volatile("vfmul.vf   v8, v8, %0"  :: "f"(momentum));
        asm volatile("vfnmsac.vf v8, %0, v16" :: "f"(lr));
        // clamp to [-1, 1]
        asm volatile("vfmax.vf v8, v8, %0" :: "f"(clip_lo));
        asm volatile("vfmin.vf v8, v8, %0" :: "f"(clip_hi));
        // w += v
        asm volatile("vfadd.vv v24, v24, v8");

        asm volatile("vse32.v v8,  (%0)" :: "r"(v) : "memory");
        asm volatile("vse32.v v24, (%0)" :: "r"(w) : "memory");

        w  += vl;
        v  += vl;
        dw += vl;
        n  -= (int)vl;
    }
}

static inline void sgd_step(float *w, float *v, const float *dw, int n, float wd, float lr)
{
#ifdef SGD_SCALAR
    sgd_step_scalar(w, v, dw, n, wd, lr);
#else
    sgd_step_vec(w, v, dw, n, wd, lr);
#endif
}

static void gradient_descent(alexnet *net)
{
    if (net->trainable.conv1) {
        sgd_step(net->conv1.weights, v_conv1_weights, d_conv1_weights, CONV1_W, WEIGHT_DECAY, g_lr);
        sgd_step(net->conv1.bias,    v_conv1_bias,    d_conv1_bias,    C1_CHANNELS, 0.0f, g_lr);
    }
    if (net->trainable.bn1) {
        sgd_step(net->bn1.gamma, v_bn1_gamma, d_bn1_gamma, C1_CHANNELS, 0.0f, g_lr);
        sgd_step(net->bn1.beta,  v_bn1_beta,  d_bn1_beta,  C1_CHANNELS, 0.0f, g_lr);
    }
    if (net->trainable.conv2) {
        sgd_step(net->conv2.weights, v_conv2_weights, d_conv2_weights, CONV2_W, WEIGHT_DECAY, g_lr);
        sgd_step(net->conv2.bias,    v_conv2_bias,    d_conv2_bias,    C1_CHANNELS, 0.0f, g_lr);
    }
    if (net->trainable.bn2) {
        sgd_step(net->bn2.gamma, v_bn2_gamma, d_bn2_gamma, C1_CHANNELS, 0.0f, g_lr);
        sgd_step(net->bn2.beta,  v_bn2_beta,  d_bn2_beta,  C1_CHANNELS, 0.0f, g_lr);
    }
    if (net->trainable.conv3) {
        sgd_step(net->conv3.weights, v_conv3_weights, d_conv3_weights, CONV3_W, WEIGHT_DECAY, g_lr);
        sgd_step(net->conv3.bias,    v_conv3_bias,    d_conv3_bias,    C2_CHANNELS, 0.0f, g_lr);
    }
    if (net->trainable.bn3) {
        sgd_step(net->bn3.gamma, v_bn3_gamma, d_bn3_gamma, C2_CHANNELS, 0.0f, g_lr);
        sgd_step(net->bn3.beta,  v_bn3_beta,  d_bn3_beta,  C2_CHANNELS, 0.0f, g_lr);
    }
    if (net->trainable.conv4) {
        sgd_step(net->conv4.weights, v_conv4_weights, d_conv4_weights, CONV4_W, WEIGHT_DECAY, g_lr);
        sgd_step(net->conv4.bias,    v_conv4_bias,    d_conv4_bias,    C2_CHANNELS, 0.0f, g_lr);
    }
    if (net->trainable.bn4) {
        sgd_step(net->bn4.gamma, v_bn4_gamma, d_bn4_gamma, C2_CHANNELS, 0.0f, g_lr);
        sgd_step(net->bn4.beta,  v_bn4_beta,  d_bn4_beta,  C2_CHANNELS, 0.0f, g_lr);
    }
    if (net->trainable.fc1) {
        sgd_step(net->fc1.weights, v_fc1_weights, d_fc1_weights, FC1_W, WEIGHT_DECAY, g_lr);
        sgd_step(net->fc1.bias,    v_fc1_bias,    d_fc1_bias,    FC1_LAYER, 0.0f, g_lr);
    }
    if (net->trainable.bn5) {
        sgd_step(net->bn5.gamma, v_bn5_gamma, d_bn5_gamma, FC1_LAYER, 0.0f, g_lr);
        sgd_step(net->bn5.beta,  v_bn5_beta,  d_bn5_beta,  FC1_LAYER, 0.0f, g_lr);
    }
    if (net->trainable.fc2) {
        sgd_step(net->fc2.weights, v_fc2_weights, d_fc2_weights, FC2_W, WEIGHT_DECAY, g_lr);
        sgd_step(net->fc2.bias,    v_fc2_bias,    d_fc2_bias,    OUT_LAYER, 0.0f, g_lr);
    }

#ifdef DUMP_TRACE
    // One-shot checksum of the POST-update weights and velocities. The [BW]
    // traces stop at the gradients, so this is the only thing that can tell
    // the scalar and vector sgd_step apart.
    {
        static int wt_done = 0;
        if (!wt_done) {
            wt_done = 1;
            #define WTRACE(nm, p_, n_) bw_trace("W:" nm, (p_), (n_))
            WTRACE("conv1.w", net->conv1.weights, CONV1_W);
            WTRACE("conv1.v", v_conv1_weights,    CONV1_W);
            WTRACE("conv4.w", net->conv4.weights, CONV4_W);
            WTRACE("bn1.g",   net->bn1.gamma,     C1_CHANNELS);
            WTRACE("bn1.gv",  v_bn1_gamma,        C1_CHANNELS);
            WTRACE("fc1.w",   net->fc1.weights,   FC1_W);
            WTRACE("fc1.v",   v_fc1_weights,      FC1_W);
            WTRACE("fc2.w",   net->fc2.weights,   FC2_W);
            WTRACE("fc2.b",   net->fc2.bias,      OUT_LAYER);
            #undef WTRACE
        }
    }
#endif
}

// --------------------------------------------------------------------------
// Gradient buffer binding.
//
// Bound per step, zeroing every gradient buffer so gradients never accumulate
// across batches. The conv and fc kernels accumulate into d_weights and
// d_input with +=, and max-pool accumulates into d_input, so those buffers
// must start at zero.
// --------------------------------------------------------------------------

static void bind_d_params(alexnet *net)
{
    net->conv1.d_weights = d_conv1_weights; net->conv1.d_bias = d_conv1_bias;
    net->conv2.d_weights = d_conv2_weights; net->conv2.d_bias = d_conv2_bias;
    net->conv3.d_weights = d_conv3_weights; net->conv3.d_bias = d_conv3_bias;
    net->conv4.d_weights = d_conv4_weights; net->conv4.d_bias = d_conv4_bias;
    net->fc1.d_weights   = d_fc1_weights;   net->fc1.d_bias   = d_fc1_bias;
    net->fc2.d_weights   = d_fc2_weights;   net->fc2.d_bias   = d_fc2_bias;

    net->bn1.d_gamma = d_bn1_gamma; net->bn1.d_beta = d_bn1_beta;
    net->bn2.d_gamma = d_bn2_gamma; net->bn2.d_beta = d_bn2_beta;
    net->bn3.d_gamma = d_bn3_gamma; net->bn3.d_beta = d_bn3_beta;
    net->bn4.d_gamma = d_bn4_gamma; net->bn4.d_beta = d_bn4_beta;
    net->bn5.d_gamma = d_bn5_gamma; net->bn5.d_beta = d_bn5_beta;

    zero_f32(d_conv1_weights, CONV1_W); zero_f32(d_conv1_bias, C1_CHANNELS);
    zero_f32(d_conv2_weights, CONV2_W); zero_f32(d_conv2_bias, C1_CHANNELS);
    zero_f32(d_conv3_weights, CONV3_W); zero_f32(d_conv3_bias, C2_CHANNELS);
    zero_f32(d_conv4_weights, CONV4_W); zero_f32(d_conv4_bias, C2_CHANNELS);
    zero_f32(d_fc1_weights,   FC1_W);   zero_f32(d_fc1_bias,   FC1_LAYER);
    zero_f32(d_fc2_weights,   FC2_W);   zero_f32(d_fc2_bias,   OUT_LAYER);

    zero_f32(d_bn1_gamma, C1_CHANNELS); zero_f32(d_bn1_beta, C1_CHANNELS);
    zero_f32(d_bn2_gamma, C1_CHANNELS); zero_f32(d_bn2_beta, C1_CHANNELS);
    zero_f32(d_bn3_gamma, C2_CHANNELS); zero_f32(d_bn3_beta, C2_CHANNELS);
    zero_f32(d_bn4_gamma, C2_CHANNELS); zero_f32(d_bn4_beta, C2_CHANNELS);
    zero_f32(d_bn5_gamma, FC1_LAYER);   zero_f32(d_bn5_beta, FC1_LAYER);
}

static void unbind_d_params(alexnet *net)
{
    net->conv1.d_weights = NULL; net->conv1.d_bias = NULL;
    net->conv2.d_weights = NULL; net->conv2.d_bias = NULL;
    net->conv3.d_weights = NULL; net->conv3.d_bias = NULL;
    net->conv4.d_weights = NULL; net->conv4.d_bias = NULL;
    net->fc1.d_weights   = NULL; net->fc1.d_bias   = NULL;
    net->fc2.d_weights   = NULL; net->fc2.d_bias   = NULL;

    net->bn1.d_gamma = NULL; net->bn1.d_beta = NULL;
    net->bn2.d_gamma = NULL; net->bn2.d_beta = NULL;
    net->bn3.d_gamma = NULL; net->bn3.d_beta = NULL;
    net->bn4.d_gamma = NULL; net->bn4.d_beta = NULL;
    net->bn5.d_gamma = NULL; net->bn5.d_beta = NULL;
}

// True when any layer below fc1 still needs gradients. When false (the
// FINETUNE case) the backward chain stops at fc1.
static int conv_stack_trainable(const alexnet *net)
{
    return net->trainable.conv1 || net->trainable.conv2 ||
           net->trainable.conv3 || net->trainable.conv4 ||
           net->trainable.bn1   || net->trainable.bn2   ||
           net->trainable.bn3   || net->trainable.bn4;
}

// --------------------------------------------------------------------------
// Backward pass, then the weight update.
// Returns the mean cross-entropy loss over the batch.
// --------------------------------------------------------------------------

#ifdef DUMP_TRACE
// One-shot checksum of the FIRST backward pass, printed in backward order.
// Forward already matches the native scalar reference, so the first line here
// that disagrees names the backward kernel that diverges.
static int bw_trace_done = 0;
static void bw_trace(const char *name, const float *p, int n)
{
    float sum = 0.0f, absum = 0.0f, amax = 0.0f;
    for (int i = 0; i < n; i++) {
        float v = p[i];
        float a = v < 0.0f ? -v : v;
        sum += v;
        absum += a;
        if (a > amax) amax = a;
    }
    printf_("[BW] %-9s n=%-7d sum=%15.6f  absum=%15.6f  max=%12.6f\n",
            name, n, sum, absum, amax);
}
#define BTRACE(name, ptr, n) \
    do { if (!bw_trace_done) bw_trace((name), (ptr), (n)); } while (0)
#else
#define BTRACE(name, ptr, n) do { } while (0)
#endif

float backward_alexnet(alexnet *net, int *batch_Y)
{
    if (net->batchsize > ALEXNET_STATIC_MAX_BATCH) {
        printf_("Error: batchsize %d exceeds static max batch %d\n",
                net->batchsize, ALEXNET_STATIC_MAX_BATCH);
        exit(1);
    }

    const int B = net->batchsize;
    const int deep = conv_stack_trainable(net);

    bind_d_params(net);

    zero_f32(d_fc2_out,   B * OUT_LAYER);
    zero_f32(d_bn5_out,   B * FC1_LAYER);
    zero_f32(d_relu5_out, B * FC1_LAYER);
    zero_f32(d_fc1_out,   B * FC1_LAYER);
    zero_f32(d_pool2_out, B * POOL2_UNITS);
    if (deep) {
        zero_f32(d_bn4_out,   B * BLOCK2_UNITS);
        zero_f32(d_relu4_out, B * BLOCK2_UNITS);
        zero_f32(d_conv4_out, B * BLOCK2_UNITS);
        zero_f32(d_bn3_out,   B * BLOCK2_UNITS);
        zero_f32(d_relu3_out, B * BLOCK2_UNITS);
        zero_f32(d_conv3_out, B * BLOCK2_UNITS);
        zero_f32(d_pool1_out, B * POOL1_UNITS);
        zero_f32(d_bn2_out,   B * BLOCK1_UNITS);
        zero_f32(d_relu2_out, B * BLOCK1_UNITS);
        zero_f32(d_conv2_out, B * BLOCK1_UNITS);
        zero_f32(d_bn1_out,   B * BLOCK1_UNITS);
        zero_f32(d_relu1_out, B * BLOCK1_UNITS);
        zero_f32(d_conv1_out, B * BLOCK1_UNITS);
        zero_f32(d_image,     B * IMAGE_UNITS);
    }

    float loss = cross_entropy_loss(d_fc2_out, net->output, batch_Y,
                                    net->fc2.out_units, B);
                                    BTRACE("d_logits", d_fc2_out, B * OUT_LAYER);

    // ---- Classifier ----
    net->fc2.d_output = d_fc2_out;
    net->fc2.d_input  = d_bn5_out;
    if (net->trainable.fc2) fc_op_backward_full(&net->fc2);
    else                    fc_op_backward_input_only(&net->fc2);
    ALEXNET_LOG_LAYER("fc2 bwd\n");
    BTRACE("fc2.dW", net->fc2.d_weights, net->fc2.in_units * net->fc2.out_units);
    BTRACE("fc2.db", net->fc2.d_bias, net->fc2.out_units);
    BTRACE("d_bn5", d_bn5_out, B * FC1_LAYER);

    // dropout after the FC head (bn5 -> [drop] -> fc2)
    dropout_backward(d_bn5_out, dofc_mask, B * net->bn5.units);

    net->bn5.d_output = d_bn5_out;
    net->bn5.d_input  = d_relu5_out;
    if (net->trainable.bn5) batch_norm_op_backward_full(&net->bn5);
    else                    batch_norm_op_backward_input_only(&net->bn5);
    ALEXNET_LOG_LAYER("bn5 bwd\n");
    BTRACE("bn5.dg", net->bn5.d_gamma, net->bn5.channels);
    BTRACE("bn5.dbeta", net->bn5.d_beta, net->bn5.channels);
    BTRACE("d_relu5", d_relu5_out, B * FC1_LAYER);

    net->relu5.d_output = d_relu5_out;
    net->relu5.d_input  = d_fc1_out;
    relu_op_backward(&net->relu5);

    net->fc1.d_output = d_fc1_out;
    net->fc1.d_input  = d_pool2_out;
    if (net->trainable.fc1) fc_op_backward_full(&net->fc1);
    else                    fc_op_backward_input_only(&net->fc1);
    ALEXNET_LOG_LAYER("fc1 bwd\n");
    BTRACE("fc1.dW", net->fc1.d_weights, net->fc1.in_units * net->fc1.out_units);
    BTRACE("fc1.db", net->fc1.d_bias, net->fc1.out_units);
    BTRACE("d_pool2", d_pool2_out, B * POOL2_UNITS);

    if (!deep) {
        // Everything below fc1 is frozen: d_pool2_out has nothing to feed.
        gradient_descent(net);
        unbind_d_params(net);
#ifdef DUMP_TRACE
        bw_trace_done = 1;
#endif
        return loss;
    }

    // ---- Block 2 ----
    // dropout after pool2 (pool2 -> [drop] -> fc1)
    dropout_backward(d_pool2_out, do2_mask, B * net->pool2.out_units);

    net->pool2.d_output = d_pool2_out;
    net->pool2.d_input  = d_bn4_out;
    max_pooling_op_backward(&net->pool2);
    ALEXNET_LOG_LAYER("pool2 bwd\n");
    BTRACE("d_bn4", d_bn4_out, B * BLOCK2_UNITS);

    net->bn4.d_output = d_bn4_out;
    net->bn4.d_input  = d_relu4_out;
    if (net->trainable.bn4) batch_norm_op_backward_full(&net->bn4);
    else                    batch_norm_op_backward_input_only(&net->bn4);
    BTRACE("bn4.dg", net->bn4.d_gamma, net->bn4.channels);
    BTRACE("bn4.dbeta", net->bn4.d_beta, net->bn4.channels);
    BTRACE("d_relu4", d_relu4_out, B * BLOCK2_UNITS);

    net->relu4.d_output = d_relu4_out;
    net->relu4.d_input  = d_conv4_out;
    relu_op_backward(&net->relu4);

    net->conv4.d_output = d_conv4_out;
    net->conv4.d_input  = d_bn3_out;
    if (net->trainable.conv4) conv_op_backward_full(&net->conv4);
    else                      conv_op_backward_input_only(&net->conv4);
    ALEXNET_LOG_LAYER("conv4 bwd\n");
    BTRACE("conv4.dW", net->conv4.d_weights, net->conv4.out_channels * net->conv4.in_channels * net->conv4.kernel_size * net->conv4.kernel_size);
    BTRACE("conv4.db", net->conv4.d_bias, net->conv4.out_channels);
    BTRACE("d_bn3", d_bn3_out, B * BLOCK2_UNITS);

    net->bn3.d_output = d_bn3_out;
    net->bn3.d_input  = d_relu3_out;
    if (net->trainable.bn3) batch_norm_op_backward_full(&net->bn3);
    else                    batch_norm_op_backward_input_only(&net->bn3);
    BTRACE("bn3.dg", net->bn3.d_gamma, net->bn3.channels);
    BTRACE("bn3.dbeta", net->bn3.d_beta, net->bn3.channels);
    BTRACE("d_relu3", d_relu3_out, B * BLOCK2_UNITS);

    net->relu3.d_output = d_relu3_out;
    net->relu3.d_input  = d_conv3_out;
    relu_op_backward(&net->relu3);

    net->conv3.d_output = d_conv3_out;
    net->conv3.d_input  = d_pool1_out;
    if (net->trainable.conv3) conv_op_backward_full(&net->conv3);
    else                      conv_op_backward_input_only(&net->conv3);
    ALEXNET_LOG_LAYER("conv3 bwd\n");
    BTRACE("conv3.dW", net->conv3.d_weights, net->conv3.out_channels * net->conv3.in_channels * net->conv3.kernel_size * net->conv3.kernel_size);
    BTRACE("conv3.db", net->conv3.d_bias, net->conv3.out_channels);
    BTRACE("d_pool1", d_pool1_out, B * POOL1_UNITS);

    // ---- Block 1 ----
    // dropout after pool1 (pool1 -> [drop] -> conv3)
    dropout_backward(d_pool1_out, do1_mask, B * net->pool1.out_units);

    net->pool1.d_output = d_pool1_out;
    net->pool1.d_input  = d_bn2_out;
    max_pooling_op_backward(&net->pool1);
    ALEXNET_LOG_LAYER("pool1 bwd\n");
    BTRACE("d_bn2", d_bn2_out, B * BLOCK1_UNITS);

    net->bn2.d_output = d_bn2_out;
    net->bn2.d_input  = d_relu2_out;
    if (net->trainable.bn2) batch_norm_op_backward_full(&net->bn2);
    else                    batch_norm_op_backward_input_only(&net->bn2);
    BTRACE("bn2.dg", net->bn2.d_gamma, net->bn2.channels);
    BTRACE("bn2.dbeta", net->bn2.d_beta, net->bn2.channels);
    BTRACE("d_relu2", d_relu2_out, B * BLOCK1_UNITS);

    net->relu2.d_output = d_relu2_out;
    net->relu2.d_input  = d_conv2_out;
    relu_op_backward(&net->relu2);

    net->conv2.d_output = d_conv2_out;
    net->conv2.d_input  = d_bn1_out;
    if (net->trainable.conv2) conv_op_backward_full(&net->conv2);
    else                      conv_op_backward_input_only(&net->conv2);
    ALEXNET_LOG_LAYER("conv2 bwd\n");
    BTRACE("conv2.dW", net->conv2.d_weights, net->conv2.out_channels * net->conv2.in_channels * net->conv2.kernel_size * net->conv2.kernel_size);
    BTRACE("conv2.db", net->conv2.d_bias, net->conv2.out_channels);
    BTRACE("d_bn1", d_bn1_out, B * BLOCK1_UNITS);

    net->bn1.d_output = d_bn1_out;
    net->bn1.d_input  = d_relu1_out;
    if (net->trainable.bn1) batch_norm_op_backward_full(&net->bn1);
    else                    batch_norm_op_backward_input_only(&net->bn1);
    BTRACE("bn1.dg", net->bn1.d_gamma, net->bn1.channels);
    BTRACE("bn1.dbeta", net->bn1.d_beta, net->bn1.channels);
    BTRACE("d_relu1", d_relu1_out, B * BLOCK1_UNITS);

    net->relu1.d_output = d_relu1_out;
    net->relu1.d_input  = d_conv1_out;
    relu_op_backward(&net->relu1);

    // conv1 — the chain ends here: d_image is written but never consumed.
    net->conv1.d_output = d_conv1_out;
    net->conv1.d_input  = d_image;
    if (net->trainable.conv1) conv_op_backward_full(&net->conv1);
    else                      conv_op_backward_input_only(&net->conv1);
    ALEXNET_LOG_LAYER("conv1 bwd\n");
    BTRACE("conv1.dW", net->conv1.d_weights, net->conv1.out_channels * net->conv1.in_channels * net->conv1.kernel_size * net->conv1.kernel_size);
    BTRACE("conv1.db", net->conv1.d_bias, net->conv1.out_channels);

    gradient_descent(net);
    unbind_d_params(net);
#ifdef DUMP_TRACE
    bw_trace_done = 1;
#endif

    return loss;
}

// --------------------------------------------------------------------------
// Training loop — trains on the train split only
// --------------------------------------------------------------------------

static void zero_velocities(void)
{
    zero_f32(v_conv1_weights, CONV1_W); zero_f32(v_conv1_bias, C1_CHANNELS);
    zero_f32(v_conv2_weights, CONV2_W); zero_f32(v_conv2_bias, C1_CHANNELS);
    zero_f32(v_conv3_weights, CONV3_W); zero_f32(v_conv3_bias, C2_CHANNELS);
    zero_f32(v_conv4_weights, CONV4_W); zero_f32(v_conv4_bias, C2_CHANNELS);
    zero_f32(v_fc1_weights,   FC1_W);   zero_f32(v_fc1_bias,   FC1_LAYER);
    zero_f32(v_fc2_weights,   FC2_W);   zero_f32(v_fc2_bias,   OUT_LAYER);

    zero_f32(v_bn1_gamma, C1_CHANNELS); zero_f32(v_bn1_beta, C1_CHANNELS);
    zero_f32(v_bn2_gamma, C1_CHANNELS); zero_f32(v_bn2_beta, C1_CHANNELS);
    zero_f32(v_bn3_gamma, C2_CHANNELS); zero_f32(v_bn3_beta, C2_CHANNELS);
    zero_f32(v_bn4_gamma, C2_CHANNELS); zero_f32(v_bn4_beta, C2_CHANNELS);
    zero_f32(v_bn5_gamma, FC1_LAYER);   zero_f32(v_bn5_beta, FC1_LAYER);
}

void alexnet_train(alexnet *net, int epochs)
{
    if (net->batchsize > ALEXNET_STATIC_MAX_BATCH) {
        printf_("Error: batchsize %d exceeds static max batch %d\n",
                net->batchsize, ALEXNET_STATIC_MAX_BATCH);
        exit(1);
    }

    ensure_split();
    dataset_set_augment(AUGMENT);   // training only; eval is never augmented

    g_lr = LEARNING_RATE;
    zero_velocities();

    float *saved_input = net->input;
    net->input       = train_input_buf;
    net->is_training = 1;

    int *batch_Y = train_batch_Y_buf;
    int *preds   = train_preds_buf;

    // Drop the partial trailing batch: BN normalises with batch statistics, so
    // a short final batch would skew them.
    int steps_per_epoch = get_train_count() / net->batchsize;
    if (steps_per_epoch <= 0) steps_per_epoch = 1;
    if (ALEXNET_MAX_STEPS > 0 && steps_per_epoch > ALEXNET_MAX_STEPS)
        steps_per_epoch = ALEXNET_MAX_STEPS;

    printf_("train: lr=%.5f  momentum=%.2f  wd=%.5f  epochs=%d  steps/epoch=%d  "
            "conv stack %s\n",
            LEARNING_RATE, MOMENTUM, WEIGHT_DECAY, epochs, steps_per_epoch,
            conv_stack_trainable(net) ? "TRAINABLE" : "FROZEN");

    for (int e = 0; e < epochs; e++) {
        dataset_shuffle_train();

        float epoch_loss = 0.0f;
        int   correct    = 0;
        int   seen       = 0;

        for (int b = 0; b < steps_per_epoch; b++) {
            get_train_batch(net->batchsize, net->input, batch_Y,
                            net->conv1.in_w, net->conv1.in_h, net->conv1.in_channels);

            forward_alexnet(net);

            for (int i = 0; i < net->batchsize; i++) {
                preds[i] = argmax(net->output + i * net->fc2.out_units,
                                  net->fc2.out_units);
                if (preds[i] == batch_Y[i]) correct++;
            }
            seen += net->batchsize;

            epoch_loss += backward_alexnet(net, batch_Y);

#ifdef SHOW_PREDCITION_DETAIL
            printf_("  step %d/%d  loss %.4f\n", b + 1, steps_per_epoch,
                    epoch_loss / (b + 1));
#endif
        }

        printf_("epoch %d/%d  lr=%.6f  loss=%.4f  train acc=%.4f (%d/%d)\n",
                e + 1, epochs, g_lr, epoch_loss / steps_per_epoch,
                (float)correct / seen, correct, seen);

        // Checkpoint every epoch: a multi-hour run should survive a crash or
        // a Ctrl-C without losing everything.
        save_alexnet(net);

        if (LR_STEP_EPOCHS > 0 && (e + 1) % LR_STEP_EPOCHS == 0) {
            g_lr *= LR_STEP_GAMMA;
            printf_("  LR step decay → %.6f\n", g_lr);
        }
    }

    net->is_training = 0;
    net->input       = saved_input;
}

// --------------------------------------------------------------------------
// Evaluation — sweeps the entire held-out split
// --------------------------------------------------------------------------

void alexnet_test(alexnet *net)
{
    if (net->batchsize > ALEXNET_STATIC_MAX_BATCH) {
        printf_("Error: batchsize %d exceeds static max batch %d\n",
                net->batchsize, ALEXNET_STATIC_MAX_BATCH);
        exit(1);
    }

    ensure_split();

    int eval_n = get_eval_count();
    if (eval_n <= 0) {
        printf_("no eval split — nothing to test\n");
        return;
    }

    float *saved_input    = net->input;
    short  saved_is_train = net->is_training;
    net->input       = test_input_buf;
    net->is_training = 0;

    int *batch_Y = test_batch_Y_buf;
    int *preds   = test_preds_buf;

    memset(metrics_true_pos,  0, OUT_LAYER * sizeof(int));
    memset(metrics_false_pos, 0, OUT_LAYER * sizeof(int));
    memset(metrics_false_neg, 0, OUT_LAYER * sizeof(int));

    eval_reset();

    int total_correct = 0;
    int total_seen    = 0;
    int steps         = (eval_n + net->batchsize - 1) / net->batchsize;

    for (int b = 0; b < steps; b++) {
        // A partial trailing batch is padded by the loader; only the first
        // `valid` predictions correspond to real eval samples.
        int valid = get_eval_batch(net->batchsize, net->input, batch_Y,
                                   net->conv1.in_w, net->conv1.in_h,
                                   net->conv1.in_channels);
        if (valid <= 0) break;

        forward_alexnet(net);

        for (int i = 0; i < valid; i++) {
            preds[i] = argmax(net->output + i * net->fc2.out_units,
                              net->fc2.out_units);

            if (preds[i] == batch_Y[i]) {
                total_correct++;
                metrics_true_pos[batch_Y[i]]++;
            } else {
                metrics_false_pos[preds[i]]++;
                metrics_false_neg[batch_Y[i]]++;
            }
        }
        total_seen += valid;
    }

    float f1_sum      = 0.0f;
    int   class_count = 0;
    for (int c = 0; c < OUT_LAYER; c++) {
        int tp = metrics_true_pos[c], fp = metrics_false_pos[c], fn = metrics_false_neg[c];
        if (tp + fp + fn > 0) {
            float prec = (tp + fp > 0) ? (float)tp / (tp + fp) : 0.0f;
            float rec  = (tp + fn > 0) ? (float)tp / (tp + fn) : 0.0f;
            f1_sum += (prec + rec > 0.0f) ? 2.0f * prec * rec / (prec + rec) : 0.0f;
            class_count++;
        }
    }

    printf_("eval: %d / %d correct  acc=%.4f  macro-F1=%.4f (over %d classes)\n",
            total_correct, total_seen, (float)total_correct / total_seen,
            (class_count > 0) ? f1_sum / class_count : 0.0f, class_count);

    net->input       = saved_input;
    net->is_training = saved_is_train;
}

// --------------------------------------------------------------------------
// Per-batch classification metrics
// --------------------------------------------------------------------------

void compute_batch_metrics(const int *preds, const int *labels, int batchsize)
{
    int correct = 0;
    for (int i = 0; i < batchsize; i++)
        if (preds[i] == labels[i]) correct++;
    printf_("batch acc: %.4f (%d/%d)\n",
            (float)correct / batchsize, correct, batchsize);

    memset(metrics_true_pos,  0, OUT_LAYER * sizeof(int));
    memset(metrics_false_pos, 0, OUT_LAYER * sizeof(int));
    memset(metrics_false_neg, 0, OUT_LAYER * sizeof(int));

    for (int i = 0; i < batchsize; i++) {
        if (preds[i] < 0 || preds[i] >= OUT_LAYER ||
            labels[i] < 0 || labels[i] >= OUT_LAYER) {
            printf_("[WARNING] invalid pred=%d label=%d\n", preds[i], labels[i]);
            continue;
        }
        if (preds[i] == labels[i]) metrics_true_pos[labels[i]]++;
        else { metrics_false_pos[preds[i]]++; metrics_false_neg[labels[i]]++; }
    }

    float f1_sum      = 0.0f;
    int   class_count = 0;
    for (int c = 0; c < OUT_LAYER; c++) {
        int tp = metrics_true_pos[c], fp = metrics_false_pos[c], fn = metrics_false_neg[c];
        if (tp + fp + fn > 0) {
            float prec = (tp + fp > 0) ? (float)tp / (tp + fp) : 0.0f;
            float rec  = (tp + fn > 0) ? (float)tp / (tp + fn) : 0.0f;
            f1_sum += (prec + rec > 0.0f) ? 2.0f * prec * rec / (prec + rec) : 0.0f;
            class_count++;
        }
    }
    printf_("batch F1:  %.4f (over %d classes)\n",
            (class_count > 0) ? f1_sum / class_count : 0.0f, class_count);
}
