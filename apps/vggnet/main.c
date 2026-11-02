//
// File:        main.c
// Description: MiniVGGNet — forward pass, setup, weight init and entry point
//
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include "kernel/alexnet.h"
#include "kernel/utils.h"
#include "kernel/weights.h"
#include "kernel/image_inference.h"
#include "kernel/cifar10_dataset.h"
#ifdef SPIKE
#include <printf.h>
#elif defined ARA_LINUX
#include <stdio.h>
#else
#include "printf.h"
#endif

#if !defined(ALEXNET_MODE_TRAIN) && !defined(ALEXNET_MODE_INFERENCE)
#define ALEXNET_MODE_TRAIN
#endif

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

// Checkpoint written by training, read back by the fine-tuning build.
#ifndef ALEXNET_CHECKPOINT
#define ALEXNET_CHECKPOINT "minivgg.weights"
#endif

// --------------------------------------------------------------------------
// Metrics helpers
// --------------------------------------------------------------------------

static int metrics_totPred[OUT_LAYER];
static int metrics_totLabel[OUT_LAYER];
static int metrics_TP[OUT_LAYER];

void metrics(float *ret, int *preds, int *labels,
             int classes, int totNum, int type)
{
    if (classes > OUT_LAYER || classes <= 0) { *ret = 0.0f; return; }

    int *totPred  = metrics_totPred;
    int *totLabel = metrics_totLabel;
    int *TP       = metrics_TP;
    memset(totPred,  0, (size_t)classes * sizeof(int));
    memset(totLabel, 0, (size_t)classes * sizeof(int));
    memset(TP,       0, (size_t)classes * sizeof(int));

    for (int p = 0; p < totNum; p++) {
        totPred[preds[p]]++;
        totLabel[labels[p]]++;
        if (preds[p] == labels[p]) TP[preds[p]]++;
    }

    int tmp_a = 0;
    for (int p = 0; p < classes; p++) tmp_a += TP[p];
    float accuracy = tmp_a * 1.0f / totNum;

    if (type == METRIC_ACCURACY)  { *ret = accuracy; return; }

    float precisions[classes];
    float macro_p = 0;
    for (int p = 0; p < classes; p++) {
        precisions[p] = TP[p] / (float)totLabel[p];
        macro_p += precisions[p];
    }
    macro_p /= classes;
    if (type == METRIC_PRECISION) { *ret = macro_p; return; }

    float recalls[classes];
    float macro_r = 0;
    for (int p = 0; p < classes; p++) {
        recalls[p] = TP[p] / (float)totPred[p];
        macro_r += recalls[p];
    }
    macro_r /= classes;
    if (type == METRIC_RECALL) { *ret = macro_r; return; }

    if (type == METRIC_F1SCORE)
        *ret = 2 * macro_p * macro_r / (macro_p + macro_r);
}

int argmax(float *arr, int n)
{
    int   idx = -1;
    float max = -1111111111.0f;
    for (int p = 0; p < n; p++) {
        if (arr[p] > max) { idx = p; max = arr[p]; }
    }
    return idx;
}

// --------------------------------------------------------------------------
// Static activation buffers
//
// The backward pass re-reads each layer's input (relu needs it for its mask,
// conv recomputes im2col from it, pool re-finds its argmax), so every forward
// activation stays resident for the whole step.
// --------------------------------------------------------------------------

#define BLOCK1_UNITS (C1_CHANNELS * FEATURE1_L * FEATURE1_L)
#define POOL1_UNITS  (C1_CHANNELS * POOLING1_L * POOLING1_L)
#define BLOCK2_UNITS (C2_CHANNELS * FEATURE2_L * FEATURE2_L)
#define POOL2_UNITS  (C2_CHANNELS * POOLING2_L * POOLING2_L)

// Block 1: conv1 -> relu1 -> bn1 -> conv2 -> relu2 -> bn2 -> pool1
static float act_conv1[ALEXNET_STATIC_MAX_BATCH * BLOCK1_UNITS];
static float act_relu1[ALEXNET_STATIC_MAX_BATCH * BLOCK1_UNITS];
static float act_bn1  [ALEXNET_STATIC_MAX_BATCH * BLOCK1_UNITS];
static float act_conv2[ALEXNET_STATIC_MAX_BATCH * BLOCK1_UNITS];
static float act_relu2[ALEXNET_STATIC_MAX_BATCH * BLOCK1_UNITS];
static float act_bn2  [ALEXNET_STATIC_MAX_BATCH * BLOCK1_UNITS];
static float act_pool1[ALEXNET_STATIC_MAX_BATCH * POOL1_UNITS];

// Block 2: conv3 -> relu3 -> bn3 -> conv4 -> relu4 -> bn4 -> pool2
static float act_conv3[ALEXNET_STATIC_MAX_BATCH * BLOCK2_UNITS];
static float act_relu3[ALEXNET_STATIC_MAX_BATCH * BLOCK2_UNITS];
static float act_bn3  [ALEXNET_STATIC_MAX_BATCH * BLOCK2_UNITS];
static float act_conv4[ALEXNET_STATIC_MAX_BATCH * BLOCK2_UNITS];
static float act_relu4[ALEXNET_STATIC_MAX_BATCH * BLOCK2_UNITS];
static float act_bn4  [ALEXNET_STATIC_MAX_BATCH * BLOCK2_UNITS];
static float act_pool2[ALEXNET_STATIC_MAX_BATCH * POOL2_UNITS];

// Classifier: fc1 -> relu5 -> bn5 -> fc2
static float act_fc1  [ALEXNET_STATIC_MAX_BATCH * FC1_LAYER];
static float act_relu5[ALEXNET_STATIC_MAX_BATCH * FC1_LAYER];
static float act_bn5  [ALEXNET_STATIC_MAX_BATCH * FC1_LAYER];
static float act_fc2  [ALEXNET_STATIC_MAX_BATCH * OUT_LAYER];

// Max-pool argmax positions, written by the vectorized max_pooling forward and
// consumed by its backward to route each gradient to the winning input. Encoded
// per output element as 0..3 = (row0,col0) (row0,col0+1) (row1,col0) (row1,col0+1).
// max_pooling_op::max_indices was declared but never pointed anywhere -- `net` is
// a static, so it was NULL and the vectorized forward stored through it.
static int pool1_max_idx[ALEXNET_STATIC_MAX_BATCH * POOL1_UNITS];
static int pool2_max_idx[ALEXNET_STATIC_MAX_BATCH * POOL2_UNITS];

// Dropout masks (inverted-dropout scale factors), one per element. Filled by
// the forward pass when training, reused by the backward pass in train.c.
// Placed after pool1, pool2 and the FC head per the reference MiniVGGNet table.
float do1_mask[ALEXNET_STATIC_MAX_BATCH * POOL1_UNITS];   // after pool1
float do2_mask[ALEXNET_STATIC_MAX_BATCH * POOL2_UNITS];   // after pool2
float dofc_mask[ALEXNET_STATIC_MAX_BATCH * FC1_LAYER];    // after bn5

void __libc_init_array(void) {}
void __libc_fini_array(void) {}

#ifdef SPIKE
extern volatile uint64_t tohost;
uintptr_t handle_trap(uintptr_t cause, uintptr_t epc, uintptr_t regs[32])
{
    (void)epc; (void)regs;
    uintptr_t code = 0x200 + (cause & 0x3ff);
    tohost = (code << 1) | 1;
    while (1) {}
}
#endif

#ifdef DUMP_TRACE
// One-shot per-layer checksum of the FIRST training forward pass. With the RNG,
// data and init matched between the native scalar build and the Spike vector
// build, every line must agree; the first line that does not is the layer whose
// kernel diverges. sum catches sign/ordering errors, absum catches cancellation
// hiding a difference, max|.| catches a single blown-up element.
static int trace_done = 0;
static void trace_layer(const char *name, const float *p, int n)
{
    float sum = 0.0f, absum = 0.0f, amax = 0.0f;
    for (int i = 0; i < n; i++) {
        float v = p[i];
        float a = v < 0.0f ? -v : v;
        sum += v;
        absum += a;
        if (a > amax) amax = a;
    }
    printf_("[TRACE] %-6s n=%-7d sum=%15.5f  absum=%15.5f  max=%12.5f\n",
            name, n, sum, absum, amax);
}
#define TRACE(name, ptr, n) \
    do { if (net->is_training && !trace_done) trace_layer((name), (ptr), (n)); } while (0)
#else
#define TRACE(name, ptr, n) do { } while (0)
#endif

// --------------------------------------------------------------------------
// Forward pass — CONV -> ACT -> BN in each block, no dropout
// --------------------------------------------------------------------------

void forward_alexnet(alexnet *net)
{
    if (net->batchsize > ALEXNET_STATIC_MAX_BATCH) {
        printf_("Error: batchsize %d exceeds static max batch %d\n",
                net->batchsize, ALEXNET_STATIC_MAX_BATCH);
        exit(1);
    }

    // BatchNorm uses batch stats + updates running stats while training, and
    // the frozen running stats at inference. Mirror the net's mode onto each.
    net->bn1.is_training = net->bn2.is_training = net->bn3.is_training =
        net->bn4.is_training = net->bn5.is_training = net->is_training;

    // ---- Block 1 ----
    TRACE("w_c1", net->conv1.weights, net->conv1.in_channels * net->conv1.kernel_size * net->conv1.kernel_size * net->conv1.out_channels);
    TRACE("b_c1", net->conv1.bias, net->conv1.out_channels);
    TRACE("input", net->input, net->batchsize * net->conv1.in_channels * net->conv1.in_w * net->conv1.in_h);
    TRACE("in_i0", net->input, net->conv1.in_units);
    net->conv1.input  = net->input;
    net->conv1.output = act_conv1;
    conv_op_forward(&net->conv1);
    TRACE("conv1", net->conv1.output, net->batchsize * net->conv1.out_channels * net->conv1.out_w * net->conv1.out_h);
    TRACE("c1_i0", net->conv1.output + 0 * net->conv1.out_units, net->conv1.out_units);
    TRACE("c1_i1", net->conv1.output + 1 * net->conv1.out_units, net->conv1.out_units);
    TRACE("c1_i2", net->conv1.output + 2 * net->conv1.out_units, net->conv1.out_units);
    TRACE("c1_i3", net->conv1.output + 3 * net->conv1.out_units, net->conv1.out_units);
    ALEXNET_LOG_LAYER("conv1 fwd\n");

#ifdef DUMP_CONV1
    // One-shot diagnostic: dump conv1 output for image 0 of the first training
    // batch. With matched RNG/data the two builds see the same image, so the
    // real conv kernels (native scalar vs Spike fconv3d) can be compared directly.
    if (net->is_training) {
        static int conv1_dumped = 0;
        if (!conv1_dumped) {
            conv1_dumped = 1;
            const int W = net->conv1.out_w, H = net->conv1.out_h;
            const int OC = net->conv1.out_channels;
            const float *o = net->conv1.output;   // image 0 starts at index 0
            float sum = 0.0f, amax = 0.0f;
            for (int i = 0; i < OC * W * H; i++) {
                sum += o[i];
                float a = o[i] < 0 ? -o[i] : o[i];
                if (a > amax) amax = a;
            }
            printf_("[DUMP_CONV1] img0 OC=%d %dx%d  sum=%.5f  max|.|=%.5f\n", OC, W, H, sum, amax);
            printf_("[DUMP_CONV1] channel 0, top-left 6x6 (idx=y*W+x):\n");
            for (int y = 0; y < 6; y++) {
                for (int x = 0; x < 6; x++)
                    printf_(" %9.4f", o[y * W + x]);
                printf_("\n");
            }
        }
    }
#endif

    net->relu1.input  = net->conv1.output;
    net->relu1.output = act_relu1;
    relu_op_forward(&net->relu1);
    TRACE("relu1", net->relu1.output, net->batchsize * net->conv1.out_channels * net->conv1.out_w * net->conv1.out_h);

    net->bn1.input  = net->relu1.output;
    net->bn1.output = act_bn1;
    batch_norm_op_forward(&net->bn1);
    TRACE("bn1", net->bn1.output, net->batchsize * net->bn1.channels * net->bn1.spatial_size);
    ALEXNET_LOG_LAYER("bn1 fwd\n");

    net->conv2.input  = net->bn1.output;
    net->conv2.output = act_conv2;
    conv_op_forward(&net->conv2);
    TRACE("conv2", net->conv2.output, net->batchsize * net->conv2.out_channels * net->conv2.out_w * net->conv2.out_h);
    ALEXNET_LOG_LAYER("conv2 fwd\n");

    net->relu2.input  = net->conv2.output;
    net->relu2.output = act_relu2;
    relu_op_forward(&net->relu2);
    TRACE("relu2", net->relu2.output, net->batchsize * net->conv2.out_channels * net->conv2.out_w * net->conv2.out_h);

    net->bn2.input  = net->relu2.output;
    net->bn2.output = act_bn2;
    batch_norm_op_forward(&net->bn2);
    TRACE("bn2", net->bn2.output, net->batchsize * net->bn2.channels * net->bn2.spatial_size);
    ALEXNET_LOG_LAYER("bn2 fwd\n");

    net->pool1.input  = net->bn2.output;
    net->pool1.output = act_pool1;
    net->pool1.max_indices = pool1_max_idx;
    max_pooling_op_forward(&net->pool1);
    TRACE("pool1", net->pool1.output, net->batchsize * net->pool1.out_units);
    ALEXNET_LOG_LAYER("pool1 fwd\n");

    if (net->is_training)
        dropout_forward(net->pool1.output, do1_mask, DROPOUT1,
                        net->batchsize * net->pool1.out_units);

    // ---- Block 2 ----
    net->conv3.input  = net->pool1.output;
    net->conv3.output = act_conv3;
    conv_op_forward(&net->conv3);
    TRACE("conv3", net->conv3.output, net->batchsize * net->conv3.out_channels * net->conv3.out_w * net->conv3.out_h);
    ALEXNET_LOG_LAYER("conv3 fwd\n");

    net->relu3.input  = net->conv3.output;
    net->relu3.output = act_relu3;
    relu_op_forward(&net->relu3);
    TRACE("relu3", net->relu3.output, net->batchsize * net->conv3.out_channels * net->conv3.out_w * net->conv3.out_h);

    net->bn3.input  = net->relu3.output;
    net->bn3.output = act_bn3;
    batch_norm_op_forward(&net->bn3);
    TRACE("bn3", net->bn3.output, net->batchsize * net->bn3.channels * net->bn3.spatial_size);
    ALEXNET_LOG_LAYER("bn3 fwd\n");

    net->conv4.input  = net->bn3.output;
    net->conv4.output = act_conv4;
    conv_op_forward(&net->conv4);
    TRACE("conv4", net->conv4.output, net->batchsize * net->conv4.out_channels * net->conv4.out_w * net->conv4.out_h);
    ALEXNET_LOG_LAYER("conv4 fwd\n");

    net->relu4.input  = net->conv4.output;
    net->relu4.output = act_relu4;
    relu_op_forward(&net->relu4);
    TRACE("relu4", net->relu4.output, net->batchsize * net->conv4.out_channels * net->conv4.out_w * net->conv4.out_h);

    net->bn4.input  = net->relu4.output;
    net->bn4.output = act_bn4;
    batch_norm_op_forward(&net->bn4);
    TRACE("bn4", net->bn4.output, net->batchsize * net->bn4.channels * net->bn4.spatial_size);
    ALEXNET_LOG_LAYER("bn4 fwd\n");

    net->pool2.input  = net->bn4.output;
    net->pool2.output = act_pool2;
    net->pool2.max_indices = pool2_max_idx;
    max_pooling_op_forward(&net->pool2);
    TRACE("pool2", net->pool2.output, net->batchsize * net->pool2.out_units);
    ALEXNET_LOG_LAYER("pool2 fwd\n");

    if (net->is_training)
        dropout_forward(net->pool2.output, do2_mask, DROPOUT2,
                        net->batchsize * net->pool2.out_units);

    // ---- Classifier ----
    // pool2's output is already contiguous per sample, so the flatten is a no-op.
    net->fc1.input  = net->pool2.output;
    net->fc1.output = act_fc1;
    fc_op_forward(&net->fc1);
    TRACE("fc1", net->fc1.output, net->batchsize * net->fc1.out_units);
    ALEXNET_LOG_LAYER("fc1 fwd\n");

    net->relu5.input  = net->fc1.output;
    net->relu5.output = act_relu5;
    relu_op_forward(&net->relu5);
    TRACE("relu5", net->relu5.output, net->batchsize * net->fc1.out_units);

    net->bn5.input  = net->relu5.output;
    net->bn5.output = act_bn5;
    batch_norm_op_forward(&net->bn5);
    TRACE("bn5", net->bn5.output, net->batchsize * net->bn5.channels * net->bn5.spatial_size);
    ALEXNET_LOG_LAYER("bn5 fwd\n");

    if (net->is_training)
        dropout_forward(net->bn5.output, dofc_mask, DROPOUT_FC,
                        net->batchsize * net->bn5.units);

    net->fc2.input  = net->bn5.output;
    net->fc2.output = act_fc2;
    fc_op_forward(&net->fc2);
    TRACE("fc2", net->fc2.output, net->batchsize * net->fc2.out_units);
    ALEXNET_LOG_LAYER("fc2 fwd\n");

    // fc2 emits raw logits; softmax is fused into the cross-entropy loss.
    net->output = net->fc2.output;
#ifdef DUMP_TRACE
    if (net->is_training) trace_done = 1;
#endif
}


void free_forward_activations(alexnet *net)
{
    net->conv1.output = NULL; net->relu1.output = NULL; net->bn1.output = NULL;
    net->conv2.output = NULL; net->relu2.output = NULL; net->bn2.output = NULL;
    net->pool1.output = NULL;
    net->conv3.output = NULL; net->relu3.output = NULL; net->bn3.output = NULL;
    net->conv4.output = NULL; net->relu4.output = NULL; net->bn4.output = NULL;
    net->pool2.output = NULL;
    net->fc1.output   = NULL; net->relu5.output = NULL; net->bn5.output = NULL;
    net->fc2.output   = NULL;
    net->output       = NULL;

    // x_norm/avg/var are scratch owned by batchnorm_layer.c and rebound on
    // every forward pass, so dropping the pointers here is safe.
    net->bn1.x_norm = NULL;
    net->bn2.x_norm = NULL;
    net->bn3.x_norm = NULL;
    net->bn4.x_norm = NULL;
    net->bn5.x_norm = NULL;
}


// --------------------------------------------------------------------------
// Weight pointer binding (uses arrays from weights.c)
// --------------------------------------------------------------------------

void malloc_alexnet(alexnet *net)
{
    net->conv1.weights = conv1_weights; net->conv1.bias = conv1_bias;
    net->conv2.weights = conv2_weights; net->conv2.bias = conv2_bias;
    net->conv3.weights = conv3_weights; net->conv3.bias = conv3_bias;
    net->conv4.weights = conv4_weights; net->conv4.bias = conv4_bias;
    net->fc1.weights   = fc1_weights;   net->fc1.bias   = fc1_bias;
    net->fc2.weights   = fc2_weights;   net->fc2.bias   = fc2_bias;

    net->bn1.gamma = bn1_gamma; net->bn1.beta = bn1_beta;
    net->bn2.gamma = bn2_gamma; net->bn2.beta = bn2_beta;
    net->bn3.gamma = bn3_gamma; net->bn3.beta = bn3_beta;
    net->bn4.gamma = bn4_gamma; net->bn4.beta = bn4_beta;
    net->bn5.gamma = bn5_gamma; net->bn5.beta = bn5_beta;
}

void free_alexnet(alexnet *net)
{
    net->conv1.weights = NULL; net->conv1.bias = NULL;
    net->conv2.weights = NULL; net->conv2.bias = NULL;
    net->conv3.weights = NULL; net->conv3.bias = NULL;
    net->conv4.weights = NULL; net->conv4.bias = NULL;
    net->fc1.weights   = NULL; net->fc1.bias   = NULL;
    net->fc2.weights   = NULL; net->fc2.bias   = NULL;

    net->bn1.gamma = NULL; net->bn1.beta = NULL;
    net->bn2.gamma = NULL; net->bn2.beta = NULL;
    net->bn3.gamma = NULL; net->bn3.beta = NULL;
    net->bn4.gamma = NULL; net->bn4.beta = NULL;
    net->bn5.gamma = NULL; net->bn5.beta = NULL;
}


// --------------------------------------------------------------------------
// Trainable layer control
// --------------------------------------------------------------------------

void alexnet_set_all_trainable(alexnet *net, short trainable)
{
    net->trainable.conv1 = trainable;
    net->trainable.conv2 = trainable;
    net->trainable.conv3 = trainable;
    net->trainable.conv4 = trainable;

    net->trainable.bn1 = trainable;
    net->trainable.bn2 = trainable;
    net->trainable.bn3 = trainable;
    net->trainable.bn4 = trainable;
    net->trainable.bn5 = trainable;

    net->trainable.fc1 = trainable;
    net->trainable.fc2 = trainable;
}

// Freeze the conv stack (and its BN layers); train only the classifier.
void alexnet_set_finetune_trainable(alexnet *net)
{
    alexnet_set_all_trainable(net, 0);
    net->trainable.fc1 = 1;
    net->trainable.bn5 = 1;
    net->trainable.fc2 = 1;
}

static void print_trainable_layers(const alexnet *net)
{
    int enabled = 0;
    printf_("Trainable layers:\n");
#define PRNT(name) printf_("  " #name ": %d\n", net->trainable.name); \
                   enabled += net->trainable.name ? 1 : 0
    PRNT(conv1); PRNT(conv2); PRNT(conv3); PRNT(conv4);
    PRNT(bn1); PRNT(bn2); PRNT(bn3); PRNT(bn4); PRNT(bn5);
    PRNT(fc1); PRNT(fc2);
#undef PRNT
    printf_("Total trainable layers: %d/11\n", enabled);
}

int alexnet_param_count(void)
{
    const int K = CONV_KERNEL_L * CONV_KERNEL_L;
    int n = 0;
    n += C1_CHANNELS * IN_CHANNELS * K + C1_CHANNELS;   // conv1
    n += C1_CHANNELS * C1_CHANNELS * K + C1_CHANNELS;   // conv2
    n += C2_CHANNELS * C1_CHANNELS * K + C2_CHANNELS;   // conv3
    n += C2_CHANNELS * C2_CHANNELS * K + C2_CHANNELS;   // conv4
    n += 2 * C1_CHANNELS;                               // bn1 gamma+beta
    n += 2 * C1_CHANNELS;                               // bn2
    n += 2 * C2_CHANNELS;                               // bn3
    n += 2 * C2_CHANNELS;                               // bn4
    n += FC1_IN_UNITS * FC1_LAYER + FC1_LAYER;          // fc1
    n += 2 * FC1_LAYER;                                 // bn5
    n += FC1_LAYER * OUT_LAYER + OUT_LAYER;             // fc2
    return n;
}


// --------------------------------------------------------------------------
// Network setup
// --------------------------------------------------------------------------

void setup_alexnet(alexnet *net, short batchsize)
{
    net->batchsize   = batchsize;
    net->is_training = 0;

    net->conv1.batchsize = batchsize; net->relu1.batchsize = batchsize;
    net->bn1.batchsize   = batchsize;
    net->conv2.batchsize = batchsize; net->relu2.batchsize = batchsize;
    net->bn2.batchsize   = batchsize; net->pool1.batchsize = batchsize;
    net->conv3.batchsize = batchsize; net->relu3.batchsize = batchsize;
    net->bn3.batchsize   = batchsize;
    net->conv4.batchsize = batchsize; net->relu4.batchsize = batchsize;
    net->bn4.batchsize   = batchsize; net->pool2.batchsize = batchsize;
    net->fc1.batchsize   = batchsize; net->relu5.batchsize = batchsize;
    net->bn5.batchsize   = batchsize; net->fc2.batchsize   = batchsize;

    // --- conv1: 3×32×32 → C1×32×32 ---
    net->conv1.in_channels  = IN_CHANNELS;
    net->conv1.out_channels = C1_CHANNELS;
    net->conv1.in_h  = FEATURE0_L; net->conv1.in_w  = FEATURE0_L;
    net->conv1.out_h = FEATURE1_L; net->conv1.out_w = FEATURE1_L;
    net->conv1.kernel_size = CONV_KERNEL_L;
    net->conv1.padding     = CONV_PADDING;
    net->conv1.stride      = CONV_STRIDES;
    net->conv1.in_units    = IN_CHANNELS * FEATURE0_L * FEATURE0_L;
    net->conv1.out_units   = C1_CHANNELS * FEATURE1_L * FEATURE1_L;
    net->conv1.layer_id    = 1;

    net->relu1.units = net->conv1.out_units;

    net->bn1.units        = net->conv1.out_units;
    net->bn1.channels     = C1_CHANNELS;
    net->bn1.spatial_size = FEATURE1_L * FEATURE1_L;
    net->bn1.layer_id     = 1;

    // --- conv2: C1×32×32 → C1×32×32 ---
    net->conv2.in_channels  = C1_CHANNELS;
    net->conv2.out_channels = C1_CHANNELS;
    net->conv2.in_h  = FEATURE1_L; net->conv2.in_w  = FEATURE1_L;
    net->conv2.out_h = FEATURE1_L; net->conv2.out_w = FEATURE1_L;
    net->conv2.kernel_size = CONV_KERNEL_L;
    net->conv2.padding     = CONV_PADDING;
    net->conv2.stride      = CONV_STRIDES;
    net->conv2.in_units    = C1_CHANNELS * FEATURE1_L * FEATURE1_L;
    net->conv2.out_units   = C1_CHANNELS * FEATURE1_L * FEATURE1_L;
    net->conv2.layer_id    = 2;

    net->relu2.units = net->conv2.out_units;

    net->bn2.units        = net->conv2.out_units;
    net->bn2.channels     = C1_CHANNELS;
    net->bn2.spatial_size = FEATURE1_L * FEATURE1_L;
    net->bn2.layer_id     = 2;

    // --- pool1: C1×32×32 → C1×16×16 ---
    net->pool1.channels    = C1_CHANNELS;
    net->pool1.kernel_size = POOL_KERNEL_L;
    net->pool1.stride      = POOL_STRIDES;
    net->pool1.in_h  = FEATURE1_L; net->pool1.in_w  = FEATURE1_L;
    net->pool1.out_h = POOLING1_L; net->pool1.out_w = POOLING1_L;
    net->pool1.in_units  = C1_CHANNELS * FEATURE1_L * FEATURE1_L;
    net->pool1.out_units = C1_CHANNELS * POOLING1_L * POOLING1_L;

    // --- conv3: C1×16×16 → C2×16×16 ---
    net->conv3.in_channels  = C1_CHANNELS;
    net->conv3.out_channels = C2_CHANNELS;
    net->conv3.in_h  = POOLING1_L; net->conv3.in_w  = POOLING1_L;
    net->conv3.out_h = FEATURE2_L; net->conv3.out_w = FEATURE2_L;
    net->conv3.kernel_size = CONV_KERNEL_L;
    net->conv3.padding     = CONV_PADDING;
    net->conv3.stride      = CONV_STRIDES;
    net->conv3.in_units    = net->pool1.out_units;
    net->conv3.out_units   = C2_CHANNELS * FEATURE2_L * FEATURE2_L;
    net->conv3.layer_id    = 3;

    net->relu3.units = net->conv3.out_units;

    net->bn3.units        = net->conv3.out_units;
    net->bn3.channels     = C2_CHANNELS;
    net->bn3.spatial_size = FEATURE2_L * FEATURE2_L;
    net->bn3.layer_id     = 3;

    // --- conv4: C2×16×16 → C2×16×16 ---
    net->conv4.in_channels  = C2_CHANNELS;
    net->conv4.out_channels = C2_CHANNELS;
    net->conv4.in_h  = FEATURE2_L; net->conv4.in_w  = FEATURE2_L;
    net->conv4.out_h = FEATURE2_L; net->conv4.out_w = FEATURE2_L;
    net->conv4.kernel_size = CONV_KERNEL_L;
    net->conv4.padding     = CONV_PADDING;
    net->conv4.stride      = CONV_STRIDES;
    net->conv4.in_units    = C2_CHANNELS * FEATURE2_L * FEATURE2_L;
    net->conv4.out_units   = C2_CHANNELS * FEATURE2_L * FEATURE2_L;
    net->conv4.layer_id    = 4;

    net->relu4.units = net->conv4.out_units;

    net->bn4.units        = net->conv4.out_units;
    net->bn4.channels     = C2_CHANNELS;
    net->bn4.spatial_size = FEATURE2_L * FEATURE2_L;
    net->bn4.layer_id     = 4;

    // --- pool2: C2×16×16 → C2×8×8 ---
    net->pool2.channels    = C2_CHANNELS;
    net->pool2.kernel_size = POOL_KERNEL_L;
    net->pool2.stride      = POOL_STRIDES;
    net->pool2.in_h  = FEATURE2_L; net->pool2.in_w  = FEATURE2_L;
    net->pool2.out_h = POOLING2_L; net->pool2.out_w = POOLING2_L;
    net->pool2.in_units  = C2_CHANNELS * FEATURE2_L * FEATURE2_L;
    net->pool2.out_units = C2_CHANNELS * POOLING2_L * POOLING2_L;

    // --- fc1: (C2*8*8) → FC1_LAYER ---
    net->fc1.in_units  = net->pool2.out_units;   // == FC1_IN_UNITS
    net->fc1.out_units = FC1_LAYER;
    net->fc1.layer_id  = 1;

    net->relu5.units = FC1_LAYER;

    net->bn5.units        = FC1_LAYER;
    net->bn5.channels     = FC1_LAYER;
    net->bn5.spatial_size = 1;
    net->bn5.layer_id     = 5;

    // --- fc2: FC1_LAYER → OUT_LAYER ---
    net->fc2.in_units  = FC1_LAYER;
    net->fc2.out_units = OUT_LAYER;
    net->fc2.layer_id  = 2;

    alexnet_set_all_trainable(net, 1);
}


// --------------------------------------------------------------------------
// Weight initialization
// --------------------------------------------------------------------------

// He (Kaiming) normal: std = sqrt(2 / fan_in), the standard choice for ReLU
// nets and what the PyTorch reference in scripts/minivgg_torch.py uses.
static void he_initialization(float *p, int n, int fan_in)
{
    // WEIGHT_INIT_GAIN scales the He std-dev. 1.0 = standard Kaiming/He
    // (sqrt(2/fan_in)); raise it to push the init further from zero.
#ifndef WEIGHT_INIT_GAIN
#define WEIGHT_INIT_GAIN 1.0f
#endif
    float stddv = (WEIGHT_INIT_GAIN) * sqrtf(2.0f / (float)fan_in);
    float V1 = 0, V2 = 0, S = 0, X;
    static int phase = 0;
    for (int shift = 0; shift < n; shift++) {
        if (phase == 0) {
            do {
                float U1 = (float)rand() / RAND_MAX;
                float U2 = (float)rand() / RAND_MAX;
                V1 = 2 * U1 - 1;
                V2 = 2 * U2 - 1;
                S  = V1 * V1 + V2 * V2;
            } while (S >= 1 || S == 0);
            X = V1 * sqrtf(-2 * logf(S) / S);
        } else {
            X = V2 * sqrtf(-2 * logf(S) / S);
        }
        phase = 1 - phase;
        p[shift] = stddv * X;
    }
}

static int verify_weight_array_shapes(void)
{
    int ok = 1;
#define CHECK_SHAPE(name, expected) \
    do { \
        size_t got = sizeof(name) / sizeof((name)[0]); \
        size_t exp = (size_t)(expected); \
        if (got != exp) { \
            printf_("[shape-error] %s: got=%zu expected=%zu\n", #name, got, exp); \
            ok = 0; \
        } \
    } while (0)

    const int K = CONV_KERNEL_L * CONV_KERNEL_L;
    CHECK_SHAPE(conv1_weights, C1_CHANNELS * IN_CHANNELS * K);
    CHECK_SHAPE(conv1_bias,    C1_CHANNELS);
    CHECK_SHAPE(conv2_weights, C1_CHANNELS * C1_CHANNELS * K);
    CHECK_SHAPE(conv2_bias,    C1_CHANNELS);
    CHECK_SHAPE(conv3_weights, C2_CHANNELS * C1_CHANNELS * K);
    CHECK_SHAPE(conv3_bias,    C2_CHANNELS);
    CHECK_SHAPE(conv4_weights, C2_CHANNELS * C2_CHANNELS * K);
    CHECK_SHAPE(conv4_bias,    C2_CHANNELS);

    CHECK_SHAPE(fc1_weights, FC1_IN_UNITS * FC1_LAYER);
    CHECK_SHAPE(fc1_bias,    FC1_LAYER);
    CHECK_SHAPE(fc2_weights, FC1_LAYER * OUT_LAYER);
    CHECK_SHAPE(fc2_bias,    OUT_LAYER);

    CHECK_SHAPE(bn1_gamma, C1_CHANNELS); CHECK_SHAPE(bn1_beta, C1_CHANNELS);
    CHECK_SHAPE(bn2_gamma, C1_CHANNELS); CHECK_SHAPE(bn2_beta, C1_CHANNELS);
    CHECK_SHAPE(bn3_gamma, C2_CHANNELS); CHECK_SHAPE(bn3_beta, C2_CHANNELS);
    CHECK_SHAPE(bn4_gamma, C2_CHANNELS); CHECK_SHAPE(bn4_beta, C2_CHANNELS);
    CHECK_SHAPE(bn5_gamma, FC1_LAYER);   CHECK_SHAPE(bn5_beta, FC1_LAYER);
#undef CHECK_SHAPE
    return ok;
}

void save_alexnet(alexnet *net)
{
#ifdef SPIKE
    // Spike is bare-metal with no filesystem: checkpointing is a no-op. This
    // also keeps fopen/fwrite out of the Spike link.
    (void)net;
    return;
#else
    const char *path = ALEXNET_CHECKPOINT;
    FILE *fp = fopen(path, "wb");
    if (!fp) {
        printf_("save_alexnet: cannot open %s\n", path);
        return;
    }

#define WFWRITE(ptr, n) fwrite((ptr), sizeof(float), (size_t)(n), fp)
    const int K = CONV_KERNEL_L * CONV_KERNEL_L;

    WFWRITE(net->conv1.weights, C1_CHANNELS * IN_CHANNELS * K);
    WFWRITE(net->conv1.bias,    C1_CHANNELS);
    WFWRITE(net->conv2.weights, C1_CHANNELS * C1_CHANNELS * K);
    WFWRITE(net->conv2.bias,    C1_CHANNELS);
    WFWRITE(net->conv3.weights, C2_CHANNELS * C1_CHANNELS * K);
    WFWRITE(net->conv3.bias,    C2_CHANNELS);
    WFWRITE(net->conv4.weights, C2_CHANNELS * C2_CHANNELS * K);
    WFWRITE(net->conv4.bias,    C2_CHANNELS);

    WFWRITE(net->fc1.weights, FC1_IN_UNITS * FC1_LAYER);
    WFWRITE(net->fc1.bias,    FC1_LAYER);
    WFWRITE(net->fc2.weights, FC1_LAYER * OUT_LAYER);
    WFWRITE(net->fc2.bias,    OUT_LAYER);

    WFWRITE(net->bn1.gamma, C1_CHANNELS); WFWRITE(net->bn1.beta, C1_CHANNELS);
    WFWRITE(net->bn2.gamma, C1_CHANNELS); WFWRITE(net->bn2.beta, C1_CHANNELS);
    WFWRITE(net->bn3.gamma, C2_CHANNELS); WFWRITE(net->bn3.beta, C2_CHANNELS);
    WFWRITE(net->bn4.gamma, C2_CHANNELS); WFWRITE(net->bn4.beta, C2_CHANNELS);
    WFWRITE(net->bn5.gamma, FC1_LAYER);   WFWRITE(net->bn5.beta, FC1_LAYER);

    // BN running stats (inference-time mean/var), appended after gamma/beta.
    // The PyTorch exporter writes the same trailer in the same order.
    WFWRITE(bn1_run_mean_buf, C1_CHANNELS); WFWRITE(bn1_run_var_buf, C1_CHANNELS);
    WFWRITE(bn2_run_mean_buf, C1_CHANNELS); WFWRITE(bn2_run_var_buf, C1_CHANNELS);
    WFWRITE(bn3_run_mean_buf, C2_CHANNELS); WFWRITE(bn3_run_var_buf, C2_CHANNELS);
    WFWRITE(bn4_run_mean_buf, C2_CHANNELS); WFWRITE(bn4_run_var_buf, C2_CHANNELS);
    WFWRITE(bn5_run_mean_buf, FC1_LAYER);   WFWRITE(bn5_run_var_buf, FC1_LAYER);
#undef WFWRITE

    fclose(fp);
    printf_("Weights saved to %s (%d parameters)\n", path, alexnet_param_count());
#endif /* SPIKE */
}

// Read a checkpoint written by save_alexnet(). The layout is raw fp32 in the
// same order, with no header, so the file size alone pins the architecture.
int load_alexnet_from_file(alexnet *net, const char *path)
{
#ifdef SPIKE
    // No filesystem on Spike; the checkpoint is embedded in weights.c instead.
    (void)net; (void)path;
    return 0;
#else
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        printf_("load_alexnet_from_file: cannot open %s\n", path);
        return 0;
    }

    fseek(fp, 0, SEEK_END);
    long bytes = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    // BN running stats (mean+var per channel) trail the learnable params in the
    // new format. A legacy checkpoint (no running stats) is still accepted.
    long param_bytes = (long)alexnet_param_count() * (long)sizeof(float);
    long stats_bytes = 2L * (2*C1_CHANNELS + 2*C2_CHANNELS + FC1_LAYER)
                       * (long)sizeof(float);
    int has_stats;
    if (bytes == param_bytes + stats_bytes)      has_stats = 1;
    else if (bytes == param_bytes)               has_stats = 0;
    else {
        printf_("load_alexnet_from_file: %s is %ld bytes, expected %ld (with "
                "running stats) or %ld (legacy) — architecture mismatch\n",
                path, bytes, param_bytes + stats_bytes, param_bytes);
        fclose(fp);
        return 0;
    }

    int ok = 1;
#define WFREAD(ptr, n) \
    do { \
        size_t want = (size_t)(n); \
        if (fread((ptr), sizeof(float), want, fp) != want) ok = 0; \
    } while (0)
    const int K = CONV_KERNEL_L * CONV_KERNEL_L;

    WFREAD(net->conv1.weights, C1_CHANNELS * IN_CHANNELS * K);
    WFREAD(net->conv1.bias,    C1_CHANNELS);
    WFREAD(net->conv2.weights, C1_CHANNELS * C1_CHANNELS * K);
    WFREAD(net->conv2.bias,    C1_CHANNELS);
    WFREAD(net->conv3.weights, C2_CHANNELS * C1_CHANNELS * K);
    WFREAD(net->conv3.bias,    C2_CHANNELS);
    WFREAD(net->conv4.weights, C2_CHANNELS * C2_CHANNELS * K);
    WFREAD(net->conv4.bias,    C2_CHANNELS);

    WFREAD(net->fc1.weights, FC1_IN_UNITS * FC1_LAYER);
    WFREAD(net->fc1.bias,    FC1_LAYER);
    WFREAD(net->fc2.weights, FC1_LAYER * OUT_LAYER);
    WFREAD(net->fc2.bias,    OUT_LAYER);

    WFREAD(net->bn1.gamma, C1_CHANNELS); WFREAD(net->bn1.beta, C1_CHANNELS);
    WFREAD(net->bn2.gamma, C1_CHANNELS); WFREAD(net->bn2.beta, C1_CHANNELS);
    WFREAD(net->bn3.gamma, C2_CHANNELS); WFREAD(net->bn3.beta, C2_CHANNELS);
    WFREAD(net->bn4.gamma, C2_CHANNELS); WFREAD(net->bn4.beta, C2_CHANNELS);
    WFREAD(net->bn5.gamma, FC1_LAYER);   WFREAD(net->bn5.beta, FC1_LAYER);

    if (has_stats) {
        WFREAD(bn1_run_mean_buf, C1_CHANNELS); WFREAD(bn1_run_var_buf, C1_CHANNELS);
        WFREAD(bn2_run_mean_buf, C1_CHANNELS); WFREAD(bn2_run_var_buf, C1_CHANNELS);
        WFREAD(bn3_run_mean_buf, C2_CHANNELS); WFREAD(bn3_run_var_buf, C2_CHANNELS);
        WFREAD(bn4_run_mean_buf, C2_CHANNELS); WFREAD(bn4_run_var_buf, C2_CHANNELS);
        WFREAD(bn5_run_mean_buf, FC1_LAYER);   WFREAD(bn5_run_var_buf, FC1_LAYER);
    }
#undef WFREAD

    fclose(fp);
    if (!ok) {
        printf_("load_alexnet_from_file: short read on %s\n", path);
        return 0;
    }
    if (!has_stats) {
        // Legacy checkpoint: no running stats on disk. Reset to identity so
        // inference-mode BN is at least well defined, and warn that eval will
        // be off until the net is retrained (or its stats recalibrated).
        batchnorm_reset_running_stats();
        printf_("load_alexnet_from_file: %s predates BN running stats; "
                "reset to identity — retrain to populate them for accurate inference\n",
                path);
    }
    printf_("Loaded %d parameters from %s\n", alexnet_param_count(), path);
    return 1;
#endif /* SPIKE */
}

void alexnet_init_weights(alexnet *net)
{
    if (net == NULL) return;

    if (!verify_weight_array_shapes()) {
        printf_("Fatal: weight array shape mismatch detected.\n");
        exit(1);
    }

    const int K = CONV_KERNEL_L * CONV_KERNEL_L;
    srand(42);

    he_initialization(net->conv1.weights, C1_CHANNELS * IN_CHANNELS * K, IN_CHANNELS * K);
    he_initialization(net->conv2.weights, C1_CHANNELS * C1_CHANNELS * K, C1_CHANNELS * K);
    he_initialization(net->conv3.weights, C2_CHANNELS * C1_CHANNELS * K, C1_CHANNELS * K);
    he_initialization(net->conv4.weights, C2_CHANNELS * C2_CHANNELS * K, C2_CHANNELS * K);
    he_initialization(net->fc1.weights, FC1_IN_UNITS * FC1_LAYER, FC1_IN_UNITS);
    he_initialization(net->fc2.weights, FC1_LAYER * OUT_LAYER,    FC1_LAYER);

    // BN running stats start at the identity (mean 0, var 1).
    batchnorm_reset_running_stats();

    // BN: gamma=1, beta=0
    for (int i = 0; i < C1_CHANNELS; i++) { net->bn1.gamma[i] = 1.0f; net->bn1.beta[i] = 0.0f; }
    for (int i = 0; i < C1_CHANNELS; i++) { net->bn2.gamma[i] = 1.0f; net->bn2.beta[i] = 0.0f; }
    for (int i = 0; i < C2_CHANNELS; i++) { net->bn3.gamma[i] = 1.0f; net->bn3.beta[i] = 0.0f; }
    for (int i = 0; i < C2_CHANNELS; i++) { net->bn4.gamma[i] = 1.0f; net->bn4.beta[i] = 0.0f; }
    for (int i = 0; i < FC1_LAYER;   i++) { net->bn5.gamma[i] = 1.0f; net->bn5.beta[i] = 0.0f; }

    // Biases: zero
    memset_vectorized_zero_f32(net->conv1.bias, C1_CHANNELS);
    memset_vectorized_zero_f32(net->conv2.bias, C1_CHANNELS);
    memset_vectorized_zero_f32(net->conv3.bias, C2_CHANNELS);
    memset_vectorized_zero_f32(net->conv4.bias, C2_CHANNELS);
    memset_vectorized_zero_f32(net->fc1.bias,   FC1_LAYER);
    memset_vectorized_zero_f32(net->fc2.bias,   OUT_LAYER);
}


// --------------------------------------------------------------------------
// Entry point
// --------------------------------------------------------------------------

#ifndef ALEXNET_BATCHSIZE
#define ALEXNET_BATCHSIZE 1
#endif
#ifndef ALEXNET_EPOCHS
#define ALEXNET_EPOCHS 100
#endif
#ifndef ALEXNET_INFER_IDX
#define ALEXNET_INFER_IDX -1
#endif

static void print_architecture(void)
{
    printf_("MiniVGGNet  (CONV->ACT->BN, no dropout)\n");
    printf_("  input      %d x %d x %d\n", IN_CHANNELS, FEATURE0_L, FEATURE0_L);
    printf_("  conv1/bn1  %d x %d x %d   3x3\n", C1_CHANNELS, FEATURE1_L, FEATURE1_L);
    printf_("  conv2/bn2  %d x %d x %d   3x3\n", C1_CHANNELS, FEATURE1_L, FEATURE1_L);
    printf_("  pool1      %d x %d x %d   2x2\n", C1_CHANNELS, POOLING1_L, POOLING1_L);
    printf_("  conv3/bn3  %d x %d x %d   3x3\n", C2_CHANNELS, FEATURE2_L, FEATURE2_L);
    printf_("  conv4/bn4  %d x %d x %d   3x3\n", C2_CHANNELS, FEATURE2_L, FEATURE2_L);
    printf_("  pool2      %d x %d x %d   2x2\n", C2_CHANNELS, POOLING2_L, POOLING2_L);
    printf_("  fc1/bn5    %d   (in %d)\n", FC1_LAYER, FC1_IN_UNITS);
    printf_("  fc2        %d\n", OUT_LAYER);
    printf_("  parameters %d\n", alexnet_param_count());
    printf_("  dropout    pool1=%.2f pool2=%.2f fc=%.2f\n",
            (double)DROPOUT1, (double)DROPOUT2, (double)DROPOUT_FC);
    printf_("\n");
}

int main(void)
{
    static alexnet net;

#if defined(FINETUNE_MODE)
    printf_("MiniVGGNet fine-tune — batchsize=%d  epochs=%d\n",
            ALEXNET_BATCHSIZE, ALEXNET_EPOCHS);
    print_architecture();
    setup_alexnet(&net, ALEXNET_BATCHSIZE);
    malloc_alexnet(&net);
#ifdef SPIKE
    // Spike has no filesystem. The pretrained checkpoint is embedded in
    // weights.c (converted from minivgg.weights), and malloc_alexnet already
    // bound the network directly to those arrays. So we do NOT load a file and
    // do NOT re-initialise to random — we adapt the embedded weights in place.
    // Freeze the conv stack; train only the classifier head (fc1, bn5, fc2).
    alexnet_set_finetune_trainable(&net);
    print_trainable_layers(&net);

    printf_("\n--- Adapting classifier head (CIFAR-100-C) ---\n");
    alexnet_train(&net, ALEXNET_EPOCHS);
    // No alexnet_test()/save here: a full held-out sweep and checkpointing are
    // impractical on Spike. Per-batch loss is printed by alexnet_train().
    free_alexnet(&net);
#else
    // Fine-tuning starts from the checkpoint that training wrote.
    if (!load_alexnet_from_file(&net, ALEXNET_CHECKPOINT)) {
        printf_("Fatal: fine-tuning needs a trained checkpoint. "
                "Run the training build first.\n");
        return 1;
    }

    alexnet_set_finetune_trainable(&net);
    print_trainable_layers(&net);

    printf_("\n--- Baseline accuracy (pre fine-tune) ---\n");
    alexnet_test(&net);

    printf_("\n--- Fine-tuning classifier head ---\n");
    alexnet_train(&net, ALEXNET_EPOCHS);

    printf_("\n--- Accuracy after fine-tune ---\n");
    alexnet_test(&net);

    save_alexnet(&net);
    free_alexnet(&net);
#endif /* SPIKE */

#elif defined(ALEXNET_MODE_TRAIN)
    printf_("MiniVGGNet training — batchsize=%d  epochs=%d  classes=%d\n",
            ALEXNET_BATCHSIZE, ALEXNET_EPOCHS, OUT_LAYER);
    print_architecture();
    setup_alexnet(&net, ALEXNET_BATCHSIZE);
    malloc_alexnet(&net);
    alexnet_init_weights(&net);

    // Train every layer.
    alexnet_set_all_trainable(&net, 1);
    print_trainable_layers(&net);

    printf_("\n--- Accuracy before training ---\n");
    alexnet_test(&net);

    alexnet_train(&net, ALEXNET_EPOCHS);

    printf_("\n--- Accuracy after training ---\n");
    alexnet_test(&net);

    save_alexnet(&net);
    free_alexnet(&net);

#elif defined(ALEXNET_MODE_INFERENCE)
    const unsigned char *infer_bytes = img_data;
    if (ALEXNET_INFER_IDX >= 0) {
        if (ALEXNET_INFER_IDX >= cifar10_count) {
            printf_("Error: ALEXNET_INFER_IDX %d out of range [0, %d)\n",
                    ALEXNET_INFER_IDX, cifar10_count);
            return 1;
        }
        infer_bytes = cifar10_data + cifar10_offsets[ALEXNET_INFER_IDX];
        printf_("inference sample: idx=%d label=%d\n",
                ALEXNET_INFER_IDX, cifar10_labels[ALEXNET_INFER_IDX]);
    }
    setup_alexnet(&net, 1);
    malloc_alexnet(&net);
    alexnet_init_weights(&net);
    printf_("MiniVGGNet setup finished. Running inference...\n");
    // NOTE: BN normalises with batch statistics, so a batch of 1 has zero
    // variance per channel. Single-image inference is only meaningful once
    // BN is given running statistics.
    alexnet_inference(&net, infer_bytes);
    free_alexnet(&net);

#else
    printf_("Error: define ALEXNET_MODE_TRAIN or ALEXNET_MODE_INFERENCE.\n");
    return 1;
#endif

    return 0;
}
