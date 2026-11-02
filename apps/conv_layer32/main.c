//
// File:        alexnet.c
// Description: alexnet.c
// Author:      Haris Wang
//
#include <stdlib.h>
#include <math.h>
// #include <assert.h>
#include <stdint.h>
#include "kernel/alexnet.h"
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
#define ALEXNET_STATIC_MAX_BATCH 4
#endif
#endif

static int metrics_totPred[OUT_LAYER];
static int metrics_totLabel[OUT_LAYER];
static int metrics_TP[OUT_LAYER];


static float act_conv1[ALEXNET_STATIC_MAX_BATCH * CONV1_OUT_UNITS];

// Vectorized: the scalar loop cost ~1108 cycles for 64 floats (the build uses
// -fno-vectorize, so it stayed scalar) and was the largest remaining item in the
// forward pass -- larger than the convolution itself (128 cycles).
static void zero_f32(float *buf, int n)
{
    size_t remaining = (size_t)n;
    // `vsetvli zero, zero` (rd = x0 AND rs1 = x0) KEEPS the caller's vl and only
    // changes vtype -- it does NOT select VLMAX. The splat then fills just those
    // lanes while the drain loop below stores at vl = min(n, VLMAX), writing the
    // untouched lanes out as GARBAGE instead of zeros. rd != x0 with rs1 = x0
    // requests AVL = ~0 -> vl = VLMAX, so the whole register group is zeroed.
    size_t vlmax_z;
    asm volatile("vsetvli %0, zero, e32, m8, ta, ma" : "=r"(vlmax_z));
    asm volatile("vmv.v.i v16, 0");
    while (remaining > 0) {
        size_t vl;
        asm volatile("vsetvli %0, %1, e32, m8, ta, ma" : "=r"(vl) : "r"(remaining));
        asm volatile("vse32.v v16, (%0)" :: "r"(buf));
        buf += vl;
        remaining -= vl;
    }
}

void __libc_init_array(void) {}
void __libc_fini_array(void) {}

static int verify_weight_array_shapes(void)
{
    size_t got_w = sizeof(conv1_weights) / sizeof(conv1_weights[0]);
    size_t got_b = sizeof(conv1_bias) / sizeof(conv1_bias[0]);
    size_t exp_w = (size_t)CONV1_WEIGHT_ELEMS;
    size_t exp_b = (size_t)CONV1_OUT_CHANNELS;
    return (got_w == exp_w) && (got_b == exp_b);
}

#ifdef SPIKE
extern volatile uint64_t tohost;

uintptr_t handle_trap(uintptr_t cause, uintptr_t epc, uintptr_t regs[32])
{
    uintptr_t mtval = 0;
    asm volatile ("csrr %0, mtval" : "=r"(mtval));

    // Τύπωσε τα πάντα πριν τα παρατήσεις!
    printf_("\n==================================\n");
    printf_("!!! FATAL EXCEPTION CAUGHT !!!\n");
    printf_("Cause: %lu\n", cause);
    printf_("EPC:   0x%lx\n", epc);
    printf_("MTVAL: 0x%lx\n", mtval);
    printf_("==================================\n");

    uintptr_t code = 0x100 + (cause & 0x3ff); 
    tohost = (code << 1) | 1;
    while (1) {}
}
#endif

void metrics(float *ret, int *preds, int *labels, 
                int classes, int totNum, int type)
{
    /**
     * Compute metric on 'preds' and 'labels'
     * 
     * Input:
     *      preds   [totNum]
     *      labels  [totNum]
     *      classes 
     *      totNum
     *      type    
     * Output:
     *      ret     
     * */

    if (classes > OUT_LAYER || classes <= 0) {
        *ret = 0.0f;
        return;
    }

    int *totPred  = metrics_totPred;
    int *totLabel = metrics_totLabel;
    int *TP       = metrics_TP;
    memset(totPred, 0, (size_t)classes * sizeof(int));
    memset(totLabel, 0, (size_t)classes * sizeof(int));
    memset(TP, 0, (size_t)classes * sizeof(int));

    for (int p = 0; p < totNum; p++)
    {
        totPred[preds[p]]++;
        totLabel[labels[p]]++;
        if(preds[p] == labels[p])
        {
            TP[preds[p]]++;
        }
    }

    int tmp_a=0;
    for (int p =0 ; p < classes; p++)
    {
        tmp_a += TP[p];
    }
    float accuracy = tmp_a * 1.0 / totNum;

    if (type == METRIC_ACCURACY)
    {
        *ret = accuracy;
        return;
    }

    float precisions[classes];
    float macro_p = 0;
    for (int p = 0; p < classes; p++)
    {
        precisions[p] = TP[p] / totLabel[p];
        macro_p += precisions[p];
    }
    macro_p /= classes;

    if (type == METRIC_PRECISION)
    {
        *ret = macro_p;
        return;
    }

    float recalls[classes];
    float macro_r = 0;
    for (int p = 0; p < classes; p++)
    {
        recalls[p] = TP[p] / totPred[p];
        macro_r += recalls[p];
    }
    macro_r /= classes;

    if (type == METRIC_RECALL)
    {
        *ret = macro_r;
        return;
    }

    if (type == METRIC_F1SCORE)
    {
        *ret = 2*macro_p*macro_r / (macro_p+macro_r);
        return;
    }
}

int argmax(float *arr, int n)
{
    /**
     * Return the index of max-value among arr ~ arr+n
     * 
     * Input:
     *      arr
     * Output:
     * Return:
     *      the index of max-value
     * */ 
    
    
     /*        for (int i = 0; i < net->batchsize; i++)
                    preds[i] = argmax(net->output + i * net->fc3.out_units, net->fc3.out_units);
    */

    //
    
    int   idx = -1;
    float max = -1111111111;
    for (int p = 0; p<n; p++)
    {
        if (arr[p] > max)
        {
            idx = p;
            max = arr[p];
        }
    }
    // assert(idx!=-1);
    return idx;
}


void forward_alexnet(alexnet *net)
{
    if (net->batchsize > ALEXNET_STATIC_MAX_BATCH) {
        printf_("Error: batchsize %d exceeds static max batch %d\n", net->batchsize, ALEXNET_STATIC_MAX_BATCH);
        exit(1);
    }

#ifdef SHOW_OP_TIME
    alexnet_timer_t start = {0};
    alexnet_timer_t finish = {0};
    double duration = 0.0;
#endif

#ifdef SHOW_OP_TIME
    alexnet_timer_now(&start);
#endif
    net->conv1.output = act_conv1;
    zero_f32(net->conv1.output, net->batchsize * net->conv1.out_units);
    net->conv1.input = net->input;
    conv_op_forward(&(net->conv1));
    net->output = net->conv1.output;
    ALEXNET_LOG_LAYER(" forward (&(net->conv1)) done\n");
#ifdef SHOW_OP_TIME
    alexnet_timer_now(&finish);
    duration = (finish.tv_sec - start.tv_sec);
    duration += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
#endif

}


void free_forward_activations(alexnet *net)
{
    /**
     * Free all heap buffers allocated by forward_alexnet.
     * Must be called after each forward pass when NOT followed by backward_alexnet
     * (which already frees these buffers itself).
     */
    net->conv1.output = NULL;

    net->output = NULL;
}


void malloc_alexnet(alexnet *net)
{
    net->conv1.weights = conv1_weights;
    net->conv1.bias = conv1_bias;
}

void free_alexnet(alexnet *net)
{
 
    net->conv1.weights = NULL;
    net->conv1.bias = NULL;
}

void release_alexnet_static_memory(alexnet *net)
{
    free_alexnet(net);
}

static void gauss_initialization(float *p, int n, int in_units, int out_units)
{
    float mean  = 0;
    float stddv = 0.01;

	float V1, V2, S, X;
	static int phase = 0;
    for (int shift = 0; shift < n; shift++)
    {
        if (phase == 0) {
            do {
                float U1 = (float) rand() / RAND_MAX;
                float U2 = (float) rand() / RAND_MAX;

                V1 = 2 * U1 - 1;
                V2 = 2 * U2 - 1;
                S = V1 * V1 + V2 * V2;
            } while (S >= 1 || S == 0);
    
            X = V1 * sqrt(-2 * log(S) / S);
        }else {
            X = V2 * sqrt(-2 * log(S) / S);
        }
        phase = 1 - phase;

        p[shift] = mean + stddv * X;
    }
}


//
// save trainable weights of network
//
void save_alexnet(alexnet *net)
{
    printf_("NOT SAVED WEIGHTS\n");
}


void load_alexnet(alexnet *net)
{
    /**
     * Validate shape compatibility for compile-time arrays.
     */
    (void)net;
     
    if (!verify_weight_array_shapes()) {
        printf_("Fatal: weight array shape mismatch detected.\n");
        exit(1);
    }

    printf_("Network pointers use weights.c arrays directly (no parameter buffer copy).\n");
}

void alexnet_set_all_trainable(alexnet *net, short trainable)
{
    net->trainable.conv1 = trainable;
}

static void print_trainable_layers(const alexnet *net)
{
    int enabled = 0;
    printf_("  conv1: %d\n", net->trainable.conv1); enabled += net->trainable.conv1 ? 1 : 0;

    printf_("Total trainable layers: %d/1\n", enabled);
}

void setup_alexnet(alexnet *net, short batchsize)
{
    /**
     * initialize alexnet
     * */
    net->batchsize = batchsize;
    printf_("batchsize in setup\n");
    net->conv1.batchsize = batchsize;
    net->conv1.in_channels = CONV1_IN_CHANNELS;
    net->conv1.out_channels = CONV1_OUT_CHANNELS;
    net->conv1.kernel_size = CONV1_KERNEL_L;
    net->conv1.padding = CONV1_PADDING;
    net->conv1.stride = CONV1_STRIDE;
    net->conv1.in_w = CONV1_IN_W;
    net->conv1.in_h = CONV1_IN_H;
    net->conv1.out_w = CONV1_OUT_W;
    net->conv1.out_h = CONV1_OUT_H;
    net->conv1.in_units = CONV1_IN_UNITS;
    net->conv1.out_units = CONV1_OUT_UNITS;
    net->conv1.layer_id = 5; // Keep input_col storage for backward.
    
    alexnet_set_all_trainable(net, 1);
}

void alexnet_init_weights(alexnet *net)
{
    if (net == NULL) {
        return;
    }

    /*
     * In training builds, embedded arrays may be placeholders; initialize
     * weights randomly unless embedded loading is explicitly enabled.
     */
#if defined(ALEXNET_MODE_TRAIN) || defined(INFERENCE_MODE) || defined(ALEXNET_USE_EMBEDDED_WEIGHTS)
    load_alexnet(net);
    return;
#endif

    // initialize weights for this network
    gauss_initialization(net->conv1.weights, CONV1_WEIGHT_ELEMS, net->conv1.in_units, net->conv1.out_units);

    for (int i = 0; i < CONV1_OUT_CHANNELS; i++)
        net->conv1.bias[i] = 1;
}


#ifndef ALEXNET_BATCHSIZE
#define ALEXNET_BATCHSIZE 4
#endif

#ifndef ALEXNET_EPOCHS
#define ALEXNET_EPOCHS 10
#endif

#ifndef ALEXNET_INFER_IDX
#define ALEXNET_INFER_IDX -1
#endif

int main(void)
{
    /**
     * 
     * Entrance
     * 
     * */
    static alexnet net;
    
    #if defined(ALEXNET_MODE_TRAIN)
    printf_("test to see if it compiled 14:36!\n");
    printf_("batch size: %d \n", ALEXNET_BATCHSIZE);
    printf_("epochs: %d \n", ALEXNET_EPOCHS);

    setup_alexnet(&net, ALEXNET_BATCHSIZE);
    malloc_alexnet(&net);
    alexnet_init_weights(&net);

    net.trainable.conv1 = 1;
    print_trainable_layers(&net);

    alexnet_train(&net, ALEXNET_EPOCHS);
    // release_alexnet_static_memory(&net);
    #elif defined(ALEXNET_MODE_INFERENCE)
    const unsigned char *infer_bytes = img_data;
    if (ALEXNET_INFER_IDX >= 0) {
        if (ALEXNET_INFER_IDX >= cifar10_count) {
            printf_("Error: ALEXNET_INFER_IDX %d out of range [0, %d)\n", ALEXNET_INFER_IDX, cifar10_count);
            return 1;
        }
        infer_bytes = cifar10_data + (size_t)ALEXNET_INFER_IDX * (size_t)cifar10_image_bytes;
        printf_("inference sample: CIFAR-10 idx=%d label=%d\n", ALEXNET_INFER_IDX, cifar10_labels[ALEXNET_INFER_IDX]);
    }
    setup_alexnet(&net, 1);
    malloc_alexnet(&net);
    alexnet_init_weights(&net);
    printf_("alexnet setup fininshed. Waiting for inference...\n");
    alexnet_inference(&net, infer_bytes);
    // release_alexnet_static_memory(&net);
    #else
    printf_("Error: define ALEXNET_MODE_TRAIN or ALEXNET_MODE_INFERENCE at compile time.\n");
    return 1;
    #endif

    return 0;
}
