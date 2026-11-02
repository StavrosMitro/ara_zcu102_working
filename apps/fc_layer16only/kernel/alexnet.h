//
// File:        alexnet.h
// Description: alexnet.h
// Author:      Haris Wang
//
#ifndef ALEXNET_H
#define ALEXNET_H

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#ifdef SPIKE
#include <printf.h>
#elif defined ARA_LINUX
#include <stdio.h>
#else
#include "printf.h"
#endif

#include "fc_layer.h"

//#define SHOW_PREDCITION_DETAIL
//#define SHOW_METRIC_EVALUTE
//#define SHOW_OP_TIME


//
//  Definition of model shape
//

//so that amount of kernels catching characteristics of the image
#define IN_CHANNELS 3
#define C1_CHANNELS 64
#define C2_CHANNELS 128
#define C3_CHANNELS 256
#define C4_CHANNELS 256
#define C5_CHANNELS 128

#define C1_KERNEL_L 3    // instead of 11
#define C2_KERNEL_L 3    // instead of 5
#define C3_KERNEL_L 3
#define C4_KERNEL_L 3
#define C5_KERNEL_L 3   // pointless to be just 1...

#define C1_STRIDES 1
#define C2_STRIDES 1
#define C3_STRIDES 1
#define C4_STRIDES 1
#define C5_STRIDES 1

#define C1_PADDING 1
#define C2_PADDING 1
#define C3_PADDING 1
#define C4_PADDING 1
#define C5_PADDING 1


#define FEATURE0_L 32
#define FEATURE1_L 32
#define POOLING1_L 16
#define FEATURE2_L 16
#define POOLING2_L 8
#define FEATURE3_L 8
#define FEATURE4_L 8
#define FEATURE5_L 8
#define POOLING5_L 1

#define FC6_LAYER   128
#define FC7_LAYER   512
// #define OUT_LAYER   1000 FOR IMAGENET
#define OUT_LAYER   10 // FOR TINY IMAGENET

// ===========================================================================
// FC test-layer dimensions — SINGLE SOURCE OF TRUTH. Change these to resize the
// layer; the op descriptor (main.c), the training buffers and data strides
// (train.c) and the weight-shape check all derive from them.
// NOTE: the input data / weights.c are generated separately — after changing
// these you MUST re-run scripts/generate_fc_test_vectors.py with matching
// INPUTS/OUTPUTS, or the embedded .bin strides won't line up.
// Constraints: FC_INPUT_UNITS <= FC_MAX_IN_UNITS, FC_OUTPUT_UNITS <= FC_MAX_INTERNAL.
// ===========================================================================
#ifndef FC_INPUT_UNITS
#define FC_INPUT_UNITS  128
#endif
#ifndef FC_OUTPUT_UNITS
#define FC_OUTPUT_UNITS 128
#endif

#define DROPOUT_PROB  0.0


typedef struct network {

    _Float16 *input;
    _Float16 *output;
    short batchsize;
    
   
    fc_op fc1;

    struct {
        short fc1;
    } trainable;
} alexnet;


//
//  Definiation of metric type
//
#define METRIC_ACCURACY  0
#define METRIC_PRECISION 1      // macro-precision
#define METRIC_RECALL    2      // macro-recall
#define METRIC_F1SCORE   3
#define METRIC_ROC       4

void metrics(float *ret, int *preds, int *labels, 
                int classes, int TotalNum, int type);
int argmax(_Float16 *arr, int n);


void malloc_alexnet(alexnet *net);
void free_alexnet(alexnet *net);
void bind_alexnet_static_memory(alexnet *net);
void release_alexnet_static_memory(alexnet *net);
void alexnet_set_all_trainable(alexnet *net, short trainable);

void set_alexnet(alexnet *net, short batchsize, char *weights_path);

void forward_alexnet(alexnet *net);
void free_forward_activations(alexnet *net);
void backward_alexnet_16(alexnet *net, const int *batch_Y, const _Float16 *batch_targets, float *loss_out);

void alexnet_train_16(alexnet *net, int epochs);
void alexnet_test_16(alexnet *net);
void alexnet_inference_16(alexnet *net, const unsigned char *img_bytes);
void compute_batch_metrics_16(const int *preds, const int *labels, int batchsize);

#endif // ALEXNET_H
