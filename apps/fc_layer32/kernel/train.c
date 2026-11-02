//
// File:        train.c
// Description: Implementation of functions related to training
// Author:      Haris Wang
//
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>
// #include <assert.h>
#include <string.h>
#include "alexnet.h"
#include "runtime.h"
#include "weights.h"
#ifdef SPIKE
#include <printf.h>
#elif defined ARA_LINUX
#include <stdio.h>
#else
#include "printf.h"
#endif

#define MAX(a,b) (((a) > (b)) ? (a) : (b))
#define MIN(a,b) (((a) < (b)) ? (a) : (b))

// train.c is compiled with -O0 in this app profile; bind timer calls to
// runtime counter primitives to avoid unresolved inline symbols.
static inline int64_t alexnet_cycle_count_local(void)
{
    int64_t cycle_count = 0;
    asm volatile("fence; csrr %0, cycle" : "=r"(cycle_count));
    return cycle_count;
}

// #define LEARNING_RATE 0.001
#define LEARNING_RATE 0.00001f

#ifndef ALEXNET_USE_MOMENTUM
#define ALEXNET_USE_MOMENTUM 1
#endif

#ifdef SHOW_OP_TIME
#ifndef ALEXNET_TIMER_HZ
#define ALEXNET_TIMER_HZ 1000000000ULL
#endif

#ifndef ALEXNET_USE_RDCYCLE_TIMER
#define ALEXNET_USE_RDCYCLE_TIMER 0
#endif

typedef struct {
    uint64_t tv_sec;
    uint64_t tv_nsec;
} alexnet_timer_t;

static inline void alexnet_timer_now(alexnet_timer_t *tp)
{
#if ALEXNET_USE_RDCYCLE_TIMER
    uint64_t cycles = 0;
    asm volatile ("rdcycle %0" : "=r"(cycles));
    tp->tv_sec = cycles / ALEXNET_TIMER_HZ;
    tp->tv_nsec = ((cycles % ALEXNET_TIMER_HZ) * 1000000000ULL) / ALEXNET_TIMER_HZ;
#else
    static uint64_t soft_ticks = 0;
    soft_ticks++;
    tp->tv_sec = soft_ticks / ALEXNET_TIMER_HZ;
    tp->tv_nsec = soft_ticks % ALEXNET_TIMER_HZ;
#endif
}
#endif

#ifndef ALEXNET_MAX_STEPS
#define ALEXNET_MAX_STEPS 0
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
#define ALEXNET_STATIC_MAX_BATCH 2
#endif
#endif

// FC_INPUT_UNITS / FC_OUTPUT_UNITS come from alexnet.h (single source of truth).
// They were re-#define'd here unconditionally, which overrode any -D from the
// build system and pinned this app to 128 regardless of the Makefile knob.
#ifndef FC_TOTAL_SAMPLES
#define FC_TOTAL_SAMPLES 4
#endif

#if defined(SET_CEL) && defined(SET_MSE)
#error "Define only one loss flag: SET_CEL or SET_MSE"
#endif

#if !defined(SET_CEL) && !defined(SET_MSE)
#define SET_MSE
#endif

static float d_fc1_weights[FC_INPUT_UNITS * FC_OUTPUT_UNITS];

static float d_fc1_bias[FC_OUTPUT_UNITS];
static float d_fc1_output_buf[ALEXNET_STATIC_MAX_BATCH * FC_OUTPUT_UNITS];

static float mse_targets_buf[ALEXNET_STATIC_MAX_BATCH * FC_OUTPUT_UNITS];

#define ALEXNET_MAX_BACKWARD_UNITS \
    MAX(MAX(MAX(MAX(MAX(MAX(MAX(MAX(MAX(MAX(MAX(MAX( \
        FC7_LAYER, \
        FC6_LAYER), \
        C5_CHANNELS * POOLING5_L * POOLING5_L), \
        C5_CHANNELS * FEATURE5_L * FEATURE5_L), \
        C4_CHANNELS * FEATURE5_L * FEATURE5_L), \
        C4_CHANNELS * FEATURE4_L * FEATURE4_L), \
        C3_CHANNELS * FEATURE4_L * FEATURE4_L), \
        C3_CHANNELS * FEATURE3_L * FEATURE3_L), \
        C2_CHANNELS * POOLING2_L * POOLING2_L), \
        C2_CHANNELS * FEATURE2_L * FEATURE2_L), \
        C1_CHANNELS * POOLING1_L * POOLING1_L), \
        C1_CHANNELS * FEATURE1_L * FEATURE1_L), \
        IN_CHANNELS * FEATURE0_L * FEATURE0_L)

// Shared ping-pong buffers for inter-layer gradient propagation.
static float d_grad_ping_0[ALEXNET_STATIC_MAX_BATCH * FC_INPUT_UNITS];

static float train_input_buf[ALEXNET_STATIC_MAX_BATCH * FC_INPUT_UNITS];
static int train_batch_Y_buf[ALEXNET_STATIC_MAX_BATCH];
static int train_preds_buf[ALEXNET_STATIC_MAX_BATCH];

static float test_input_buf[ALEXNET_STATIC_MAX_BATCH * FC_INPUT_UNITS];
static int test_batch_Y_buf[ALEXNET_STATIC_MAX_BATCH];
static int test_preds_buf[ALEXNET_STATIC_MAX_BATCH];

static int metrics_true_pos[FC_OUTPUT_UNITS];
static int metrics_false_pos[FC_OUTPUT_UNITS];
static int metrics_false_neg[FC_OUTPUT_UNITS];

static int64_t last_loss_cycles = 0;
static int64_t last_zero_dinput_cycles = 0;
static int64_t last_fc_backward_total_cycles = 0;
static int64_t last_update_cycles = 0;
static fc_backward_cycle_breakdown last_fc_backward_breakdown = {0, 0, 0};

static void zero_f32(float *buf, int n)
{
    memset(buf, 0, (size_t)n * sizeof(float));
}

static void zero_f32_vec(float *buf, int n)
{
    size_t max_vl;
    

    asm volatile("vsetvli %0, zero, e32, m8, ta, ma" : "=r"(max_vl));

    asm volatile("vmv.v.i v8, 0");

    float *ptr = buf;
    while (n > 0) {
        size_t vl;
        
        asm volatile("vsetvli %0, %1, e32, m8, ta, ma" : "=r"(vl) : "r"(n));

        asm volatile("vse32.v v8, (%0)" :: "r"(ptr));

        ptr += vl;
        n -= vl;
    }
}

static uint64_t checksum_bytes(const unsigned char *bytes, size_t len)
{
    uint64_t hash = 1469598103934665603ULL;
    for (size_t i = 0; i < len; i++) {
        hash ^= (uint64_t)bytes[i];
        hash *= 1099511628211ULL;
    }
    return hash;
}

static uint64_t checksum_f32(const float *arr, size_t n)
{
    return checksum_bytes((const unsigned char *)arr, n * sizeof(float));
}

static uint64_t checksum_i32(const int *arr, size_t n)
{
    return checksum_bytes((const unsigned char *)arr, n * sizeof(int));
}


static float cross_entropy_loss(float *delta_preds, const float *preds, const int *labels, int units, int BATCH_SIZE)
{
    /**
     * Cross Entropy backward
     * 
     * Input:
     *      preds       [BATCH_SIZE, units]
     *      labels      [BATCH_SIZE]
     * Output:
     *      delta_preds [BATCH_SIZE, units]  (per-sample gradients)
     * */
    float ce_loss = 0;
    for (int p = 0; p < BATCH_SIZE; p++)
    {
        // find max for numerical stability (log-sum-exp trick)
        register float max_val = preds[p*units];
        for (int i = 1; i < units; i++)
            if (preds[i+p*units] > max_val) max_val = preds[i+p*units];

        register float esum = 0;
        for (int i = 0; i < units; i++)
            esum += exp(preds[i+p*units] - max_val);

        ce_loss += 0 - log(exp(preds[labels[p]+p*units] - max_val) / esum);

        for (int i = 0; i < units; i++)
        {
            if (labels[p] == i) {
                delta_preds[p * units + i] = exp(preds[i+p*units] - max_val) / esum - 1;
            }else {
                delta_preds[p * units + i] = exp(preds[i+p*units] - max_val) / esum;
            } 
        }
    }
    ce_loss /= BATCH_SIZE;
    ALEXNET_LOG_LAYER("cross entropy loss computed\n");
    return ce_loss;
}


static float v_fc1_weights[FC_INPUT_UNITS * FC_OUTPUT_UNITS];
static float v_fc1_bias[FC_OUTPUT_UNITS];



static inline void CLIP(float *x, float down, float up)
{
    *x = MIN(up, MAX(down, *x));
}

static void momentum_sgd(float *w, float *v_w, float *d_w, int units)
{
    /**
     * momentum stochastic gradient descent
     * 
     * Input:
     *      w   [units]
     *      v_w [units]
     *      d_w [units]
     * Output:
     *      w   [units]
     *      v_w [units]
     * */     
    static int momentum_debug_once = 0;
    if (!momentum_debug_once) {
        printf_("momentum_sgd args: w=%p v=%p d=%p units=%d\n", w, v_w, d_w, units);
        momentum_debug_once = 1;
    }

    for (int i = 0; i < units; i++)
    {
#if ALEXNET_USE_MOMENTUM
        v_w[i] = 0.9f * v_w[i] - LEARNING_RATE * d_w[i];
        CLIP(v_w + i, -1.0f, 1.0f);
        w[i] = w[i] + v_w[i];
#else
        (void)v_w;
        w[i] = w[i] - LEARNING_RATE * d_w[i]; //238 in editor
#endif
    }
}

static void momentum_sgd_vec(float *w, float *v_w, const float *d_w, int units)
{
    float lr = LEARNING_RATE;
    int n = units;

#if ALEXNET_USE_MOMENTUM
    float momentum = 0.9f;
    float clip_min = -1.0f;
    float clip_max = 1.0f;

    while (n > 0) {
        size_t vl;
        
        asm volatile("vsetvli %0, %1, e32, m8, ta, ma" : "=r"(vl) : "r"(n));

        // v8: Velocity (v_w), v16: Gradients (d_w), v24: Weights (w)
        asm volatile("vle32.v v8,  (%0)" :: "r"(v_w));
        asm volatile("vle32.v v16, (%0)" :: "r"(d_w));
        asm volatile("vle32.v v24, (%0)" :: "r"(w));

        // 2. v_w = v_w * 0.9f
        asm volatile("vfmul.vf v8, v8, %0" :: "f"(momentum));

        // 3. v_w = v_w - LR * d_w
        // vfnmsac = Vector Floating-point Negative Multiply-Subtract Accumulate
        // vd = -(rs1 * vs2) + vd
        asm volatile("vfnmsac.vf v8, %0, v16" :: "f"(lr));

        // 4. CLIP(v_w, -1.0f, 1.0f)
        asm volatile("vfmax.vf v8, v8, %0" :: "f"(clip_min));
        asm volatile("vfmin.vf v8, v8, %0" :: "f"(clip_max));

        asm volatile("vfadd.vv v24, v24, v8");

        asm volatile("vse32.v v8,  (%0)" :: "r"(v_w));
        asm volatile("vse32.v v24, (%0)" :: "r"(w));

        w += vl;
        v_w += vl;
        d_w += vl;
        n -= vl;
    }
#else
    (void)v_w;
    while (n > 0) {
        size_t vl;
        asm volatile("vsetvli %0, %1, e32, m8, ta, ma" : "=r"(vl) : "r"(n));

        asm volatile("vle32.v v8,  (%0)" :: "r"(w));
        asm volatile("vle32.v v16, (%0)" :: "r"(d_w));

        // w = w - LR * d_w (Με χρήση της εντολής Negative MAC)
        asm volatile("vfnmsac.vf v8, %0, v16" :: "f"(lr));

        asm volatile("vse32.v v8,  (%0)" :: "r"(w));

        w += vl;
        d_w += vl;
        n -= vl;
    }
#endif
}


static void gradient_descent_a(void *argv)
{
    alexnet *net = (alexnet *)argv;
    if (net->trainable.fc1)
        momentum_sgd_vec(fc1_weights_32, v_fc1_weights, d_fc1_weights,
                     FC_INPUT_UNITS * FC_OUTPUT_UNITS);
}

static void gradient_descent_d(void *argv)
{
    alexnet *net = (alexnet *)argv;
    if (net->trainable.fc1)
        momentum_sgd_vec(fc1_bias_32, v_fc1_bias, d_fc1_bias,
                     FC_OUTPUT_UNITS);
}

static void gradient_descent(alexnet *net)
{
    // Keep FC1 parameter pointers anchored to static storage used by this app.
    net->fc1.weights = fc1_weights_32;
    net->fc1.bias = fc1_bias_32;
    net->fc1.d_weights = d_fc1_weights;
    net->fc1.d_bias = d_fc1_bias;

    gradient_descent_a((void *)(net));
    gradient_descent_d((void *)(net));
}


void calloc_alexnet_d_params_32(alexnet *net)
{
    net->fc1.d_weights = d_fc1_weights;
    net->fc1.d_bias = d_fc1_bias;
    zero_f32_vec(net->fc1.d_weights, net->fc1.in_units * net->fc1.out_units);
    zero_f32_vec(net->fc1.d_bias, net->fc1.out_units);
}

void free_alexnet_d_params_32(alexnet *net)
{
    net->fc1.d_weights = NULL;
    net->fc1.d_bias = NULL;
}

static float mse_loss(float *delta_preds, const float *preds, const float *targets, int units, int BATCH_SIZE)
{
    /**
     * Mean Squared Error backward
     * * Input:
     * preds       [BATCH_SIZE, units]
     * targets     [BATCH_SIZE, units]
     * Output:
     * delta_preds [BATCH_SIZE, units] (per-sample gradients)
     * */
    float mse_loss_val = 0;

    for (int p = 0; p < BATCH_SIZE; p++)
    {
        for (int i = 0; i < units; i++)
        {
            int idx = p * units + i;
            
            float diff = preds[idx] - targets[idx];
            
            delta_preds[idx] = diff; 
            
            mse_loss_val += 0.5f * diff * diff;
        }
    }
    
    mse_loss_val /= (BATCH_SIZE * units);
    ALEXNET_LOG_LAYER("MSE loss computed: %f\n", mse_loss_val);
    return mse_loss_val;
}

static float mse_loss_vec(float *delta_preds, const float *preds, const float *targets, int units, int BATCH_SIZE)
{
    int total_elems = BATCH_SIZE * units;
    
    //$\sum (0.5 \cdot d^2) = 0.5 \cdot \sum (d^2)$

    float scale_factor = 0.5f / (float)total_elems;

    size_t max_vl; //vector length * 8 basically due LMUL=8
    asm volatile("vsetvli %0, zero, e32, m8, tu, ma" : "=r"(max_vl));

    asm volatile("vmv.v.i v8, 0");

    int n = total_elems;
    const float *p_ptr = preds;
    const float *t_ptr = targets;
    float *d_ptr = delta_preds;

    while (n > 0) {
        size_t vl;
        
        asm volatile("vsetvli %0, %1, e32, m8, tu, ma" : "=r"(vl) : "r"(n)); //we have set max_vl and compare with n=total_elements

        asm volatile("vle32.v v16, (%0)" :: "r"(p_ptr));
        asm volatile("vle32.v v24, (%0)" :: "r"(t_ptr));

        asm volatile("vfsub.vv v16, v16, v24");

        asm volatile("vse32.v v16, (%0)" :: "r"(d_ptr));

        asm volatile("vfmacc.vv v8, v16, v16");

        p_ptr += vl;
        t_ptr += vl;
        d_ptr += vl;
        n -= vl;
    }

    // reduction
    

    // `vsetvli zero, zero` (rd = x0 AND rs1 = x0) does NOT select VLMAX -- it
    // KEEPS the current vl and only changes vtype. Here that vl is whatever the
    // loop's LAST iteration used, i.e. the partial tail when total_elems is not
    // a multiple of VLMAX. A reduction only reads vl elements of vs2, so every
    // accumulator lane above that tail was silently dropped and the loss came
    // out too small. rd != x0 with rs1 = x0 requests AVL = ~0 -> vl = VLMAX,
    // which is what the accumulator was zeroed and filled at.
    asm volatile("vsetvli %0, zero, e32, m8, tu, ma" : "=r"(max_vl));
    asm volatile("vmv.v.i v0, 0");                         // zero entire v0..v7 group
    asm volatile("vfredsum.vs v0, v8, v0");               // sum all elements of v8..v15 into v0[0]
    float sum_squares;
    asm volatile("vfmv.f.s %0, v0" : "=f"(sum_squares));

    float mse_loss_val = sum_squares * scale_factor;
    
    ALEXNET_LOG_LAYER("MSE loss computed: %f\n", mse_loss_val);
    return mse_loss_val;
}


void backward_alexnet_32(alexnet *net, const int *batch_Y, const float *batch_targets, float *loss_out)
{
    /**
     * alexnet backward
     * 
     * Input:
     *      net:      our network
     *      batch_Y:  labels of images
     * Output:
     *      net
     * */
    calloc_alexnet_d_params_32(net);

    if (net->batchsize > ALEXNET_STATIC_MAX_BATCH) {
        printf_("Error: batchsize %d exceeds static max batch %d\n", net->batchsize, ALEXNET_STATIC_MAX_BATCH);
        exit(1);
    }

    float *curr_grad = d_fc1_output_buf;
    float *next_grad = d_grad_ping_0;
    float loss_val = 0.0f;
    int64_t t0 = 0;

    t0 = alexnet_cycle_count_local();
#if defined(SET_CEL)
    loss_val = cross_entropy_loss(curr_grad, net->fc1.output, batch_Y, net->fc1.out_units, net->batchsize);
#else
    loss_val = mse_loss_vec(curr_grad, net->fc1.output, batch_targets, net->fc1.out_units, net->batchsize);
#endif
    last_loss_cycles = alexnet_cycle_count_local() - t0;

    net->fc1.d_input = next_grad;
    t0 = alexnet_cycle_count_local();
    zero_f32_vec(net->fc1.d_input, net->batchsize * net->fc1.in_units);
    last_zero_dinput_cycles = alexnet_cycle_count_local() - t0;
    net->fc1.d_output = curr_grad;

    // printf_("\n--- BEFORE BACKWARD ---\n");
    // printf_("Weights Pointer: 0x%lx\n", (uintptr_t)net->fc1.weights);
    // printf_("Bias Pointer:    0x%lx\n", (uintptr_t)net->fc1.bias);
    
    if (net->trainable.fc1) {
        t0 = alexnet_cycle_count_local();
        fc_op_backward_full_profile_32(&(net->fc1), &last_fc_backward_breakdown);
        last_fc_backward_total_cycles = alexnet_cycle_count_local() - t0;
    } else {
        fc_op_backward_input_only_32(&(net->fc1));
        last_fc_backward_breakdown.d_input_cycles = 0;
        last_fc_backward_breakdown.d_bias_cycles = 0;
        last_fc_backward_breakdown.d_weights_cycles = 0;
        last_fc_backward_total_cycles = 0;
    }
        
    // printf_("\n--- AFTER BACKWARD ---\n");
    // printf_("Weights Pointer: 0x%lx\n", (uintptr_t)net->fc1.weights);
    // printf_("Bias Pointer:    0x%lx\n", (uintptr_t)net->fc1.bias);

    ALEXNET_LOG_LAYER(" backward (&(net->fc1)) done\n");
    t0 = alexnet_cycle_count_local();
    gradient_descent(net);
    last_update_cycles = alexnet_cycle_count_local() - t0;

    // free_alexnet_d_params_32(net);
    if (loss_out != NULL)
        *loss_out = loss_val;
}


void alexnet_train_32(alexnet *net, int epochs)
{
    if (net->batchsize > ALEXNET_STATIC_MAX_BATCH) {
        printf_("Error: batchsize %d exceeds static max batch %d\n", net->batchsize, ALEXNET_STATIC_MAX_BATCH);
        exit(1);
    }

    net->input = train_input_buf;
    int *batch_Y = train_batch_Y_buf;
    int *preds = train_preds_buf;

    int dataset_count = FC_TOTAL_SAMPLES;
    int steps_per_epoch = dataset_count / net->batchsize;
    if (dataset_count % net->batchsize) steps_per_epoch++;
    if (steps_per_epoch <= 0) steps_per_epoch = 1;
    if (ALEXNET_MAX_STEPS > 0 && steps_per_epoch > ALEXNET_MAX_STEPS)
        steps_per_epoch = ALEXNET_MAX_STEPS;

    ALEXNET_LOG_LAYER("\n\n>>>>>>>>>>>>>>>>>>>>>>>>>>> training begin >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n");
    for (int e = 0; e < epochs; e++)
    {
        printf_(">>>>>>>>>>>>>>>>>>>> epoch %d >>>>>>>>>>>>>>>>>>>>>>\n", e + 1);
        ALEXNET_LOG_LAYER("============================= epoch %d / %d =============================\n", e+1, epochs);
        for (int b = 0; b < steps_per_epoch; b++)
        {
            float step_loss = 0.0f;
            int64_t prep_cycles = 0;
            int64_t forward_cycles = 0;
            int64_t pred_metric_cycles = 0;
            int64_t backward_wrapper_cycles = 0;
            int64_t t0 = 0;
            ALEXNET_LOG_LAYER("-----------------------------step %d / %d---------------------------------\n", b+1, steps_per_epoch);

            // get_same_batch_32(net->batchsize, net->input, batch_Y,
            //                net->conv1.in_w, net->conv1.in_h, net->conv1.in_channels, net->fc3.out_units);
            t0 = alexnet_cycle_count_local();
            int sample_offset = (b * net->batchsize) % dataset_count;
            memcpy(net->input,
                     test_inputs_32 + sample_offset * FC_INPUT_UNITS,
                     (size_t)net->batchsize * FC_INPUT_UNITS * sizeof(float));

        #if defined(SET_CEL)
            for (int i = 0; i < net->batchsize; i++)
                batch_Y[i] = test_labels_32[sample_offset + i];
        #else
            memcpy(mse_targets_buf,
                     test_targets_32 + sample_offset * FC_OUTPUT_UNITS,
                     (size_t)net->batchsize * FC_OUTPUT_UNITS * sizeof(float));
            for (int i = 0; i < net->batchsize; i++)
                batch_Y[i] = argmax((float *)(mse_targets_buf + i * FC_OUTPUT_UNITS), FC_OUTPUT_UNITS);
        #endif
            prep_cycles = alexnet_cycle_count_local() - t0;

            t0 = alexnet_cycle_count_local();
            forward_alexnet(net);
            forward_cycles = alexnet_cycle_count_local() - t0;

            t0 = alexnet_cycle_count_local();
            for (int i = 0; i < net->batchsize; i++)
                preds[i] = argmax(net->output + i * net->fc1.out_units, net->fc1.out_units);

#ifdef SHOW_PREDCITION_DETAIL
            printf_("pred[ ");
            for (int i = 0; i < net->batchsize; i++)
                printf_("%d ", preds[i]);
            printf_("]  label[ ");
            for (int i = 0; i < net->batchsize; i++)
                printf_("%d ", batch_Y[i]);
            printf_("]\n");
#endif
            compute_batch_metrics_32(preds, batch_Y, net->batchsize);
            pred_metric_cycles = alexnet_cycle_count_local() - t0;

            t0 = alexnet_cycle_count_local();
            backward_alexnet_32(net, batch_Y, mse_targets_buf, &step_loss);
            backward_wrapper_cycles = alexnet_cycle_count_local() - t0;

                printf_("cycles[epoch %d batch %d/%d]: prep=%ld, forward=%ld, pred+metric=%ld, loss=%ld, zero_d_input=%ld, backward_d_input=%ld, backward_d_bias=%ld, backward_d_weights=%ld, backward_total=%ld, update=%ld, backward_wrapper=%ld\n",
                    e + 1, b + 1, steps_per_epoch,
                    prep_cycles, forward_cycles, pred_metric_cycles,
                    last_loss_cycles, last_zero_dinput_cycles,
                    last_fc_backward_breakdown.d_input_cycles,
                    last_fc_backward_breakdown.d_bias_cycles,
                    last_fc_backward_breakdown.d_weights_cycles,
                    last_fc_backward_total_cycles,
                    last_update_cycles,
                    backward_wrapper_cycles);
            printf_("epoch %d step %d/%d loss: %.6f\n", e + 1, b + 1, steps_per_epoch, step_loss);
        }
        ALEXNET_LOG_LAYER("============================= epoch %d / %d end =============================\n", e+1, epochs);
    }
    ALEXNET_LOG_LAYER(">>>>>>>>>>>>>>>>>>>>>>>>>>> training end >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n\n");

    (void)batch_Y;
    (void)preds;
}

void alexnet_test_32(alexnet *net)
{
    if (net->batchsize > ALEXNET_STATIC_MAX_BATCH) {
        printf_("Error: batchsize %d exceeds static max batch %d\n", net->batchsize, ALEXNET_STATIC_MAX_BATCH);
        exit(1);
    }

    int *batch_Y = test_batch_Y_buf;
    int *preds   = test_preds_buf;

    // In-memory one-batch evaluation (same data source path as training).
    int steps = 1;

    printf_(">>>>>>>>>>>>>>>>>>>>>>>>>> start test pass >>>>>>>>>>>>>>>>>>>>>>>>>>>>\n\n");

    // allocate a fresh input buffer for the test pass
    float *test_input = test_input_buf;
    float *saved_input = net->input;
    net->input = test_input;

    int total_correct = 0;
    int total_seen    = 0;

    for (int b = 0; b < steps; b++)
    {
        int sample_offset = (b * net->batchsize) % FC_TOTAL_SAMPLES;
        memcpy(net->input,
               test_inputs_32 + sample_offset * FC_INPUT_UNITS,
               (size_t)net->batchsize * FC_INPUT_UNITS * sizeof(float));
#if defined(SET_CEL)
        for (int i = 0; i < net->batchsize; i++)
            batch_Y[i] = test_labels_32[sample_offset + i];
#else
        memcpy(mse_targets_buf,
               test_targets_32 + sample_offset * FC_OUTPUT_UNITS,
               (size_t)net->batchsize * FC_OUTPUT_UNITS * sizeof(float));
        for (int i = 0; i < net->batchsize; i++)
            batch_Y[i] = argmax((float *)(mse_targets_buf + i * FC_OUTPUT_UNITS), FC_OUTPUT_UNITS);
#endif

        forward_alexnet(net);
        printf_("batch %d/%d  forward done\n", b+1, steps);

        for (int i = 0; i < net->batchsize; i++)
            preds[i] = argmax(net->output + i * net->fc1.out_units, net->fc1.out_units);

        // free_forward_activations(net);   // release all layer outputs allocated by forward

#ifdef SHOW_PREDCITION_DETAIL
        printf_("pred[ ");
        for (int i = 0; i < net->batchsize; i++)
            printf_("%d ", preds[i]);
        printf_("]  label[ ");
        for (int i = 0; i < net->batchsize; i++)
            printf_("%d ", batch_Y[i]);
        printf_("]\n");
#endif

        printf_("Test batch %d/%d stats\n", b+1, steps);
        compute_batch_metrics_32(preds, batch_Y, net->batchsize);

        for (int i = 0; i < net->batchsize; i++)
            if (preds[i] == batch_Y[i]) total_correct++;
        total_seen += net->batchsize;
    }

    printf_("\n--- Overall test results: %d / %d correct  (accuracy %.4f) ---\n",
           total_correct, total_seen, (float)total_correct / total_seen);
    printf_(">>>>>>>>>>>>>>>>>>>>>>>>>>> test pass end >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n\n");

    (void)test_input;
    net->input = saved_input;
    (void)batch_Y;
    (void)preds;
}


void compute_batch_metrics_32(const int *preds, const int *labels, int batchsize)
{
    const int classes = FC_OUTPUT_UNITS;

    // Batch accuracy
    int correct = 0;
    for (int i = 0; i < batchsize; i++)
        if (preds[i] == labels[i]) correct++;
    float accuracy = (float)correct / batchsize;
#ifdef SHOW_METRIC_EVALUTE
    printf_("batch accuracy:  %.4f  (%d / %d correct)\n", accuracy, correct, batchsize);
#else
    (void)accuracy;
#endif

    int *true_pos = metrics_true_pos;
    int *false_pos = metrics_false_pos;
    int *false_neg = metrics_false_neg;

    memset(true_pos,  0, classes * sizeof(int));
    memset(false_pos, 0, classes * sizeof(int));
    memset(false_neg, 0, classes * sizeof(int));
    
    for (int i = 0; i < batchsize; i++) {
        if (preds[i] < 0 || preds[i] >= classes || labels[i] < 0 || labels[i] >= classes) {
            printf_("[WARNING] Invalid data! Pred: %d, Label: %d\n", preds[i], labels[i]);
            continue;
        }

        if (preds[i] == labels[i]) {
            true_pos[labels[i]]++;
        } else {
            false_pos[preds[i]]++;
            false_neg[labels[i]]++;
        }
    }
    
    float f1_sum = 0.0f;
    int   class_count = 0;
    for (int c = 0; c < classes; c++) {
        if (true_pos[c] + false_pos[c] + false_neg[c] > 0) {
            float prec = (true_pos[c] + false_pos[c] > 0)
                         ? (float)true_pos[c] / (true_pos[c] + false_pos[c]) : 0.0f;
            float rec  = (true_pos[c] + false_neg[c] > 0)
                         ? (float)true_pos[c] / (true_pos[c] + false_neg[c]) : 0.0f;
            float f1   = (prec + rec > 0.0f)
                         ? 2.0f * prec * rec / (prec + rec) : 0.0f;
            f1_sum += f1;
            class_count++;
        }
    }
    float macro_f1 = (class_count > 0) ? f1_sum / class_count : 0.0f;
#ifdef SHOW_METRIC_EVALUTE
    printf_("batch macro F1:  %.4f  (over %d classes)\n", macro_f1, class_count);
#else
    (void)macro_f1;
#endif
}