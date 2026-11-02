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

#ifndef ALEXNET_BATCHSIZE
#define ALEXNET_BATCHSIZE 2
#endif

#ifndef ALEXNET_STATIC_MAX_BATCH
#define ALEXNET_STATIC_MAX_BATCH ALEXNET_BATCHSIZE
#endif

#include "fc_layer.h"

#include "convolution_layer.h"

#define SHOW_PREDCITION_DETAIL
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
#define POOLING5_L 4

#define FC6_LAYER   512
#define FC7_LAYER   512
// #define OUT_LAYER   1000 FOR IMAGENET
#define OUT_LAYER   10 // FOR TINY IMAGENET

#define DROPOUT_PROB  0.0

//
// Single convolution layer training profile.
// All CONV1_* values can be overridden at compile time via -D flags.
//
#ifndef CONV1_IN_CHANNELS
#define CONV1_IN_CHANNELS  43
#endif
#ifndef CONV1_OUT_CHANNELS
#define CONV1_OUT_CHANNELS 43
#endif
#ifndef CONV1_KERNEL_L
#define CONV1_KERNEL_L 3
#endif
#ifndef CONV1_STRIDE
#define CONV1_STRIDE 1
#endif
#ifndef CONV1_PADDING
#define CONV1_PADDING 1
#endif
#ifndef CONV1_IN_W
#define CONV1_IN_W 8
#endif
#ifndef CONV1_IN_H
#define CONV1_IN_H 8
#endif
#ifndef CONV1_OUT_W
#define CONV1_OUT_W CONV1_IN_W
#endif
#ifndef CONV1_OUT_H
#define CONV1_OUT_H CONV1_IN_H
#endif
#define CONV1_IN_UNITS     (CONV1_IN_CHANNELS  * CONV1_IN_W  * CONV1_IN_H)
#define CONV1_OUT_UNITS    (CONV1_OUT_CHANNELS * CONV1_OUT_W * CONV1_OUT_H)
#define CONV1_WEIGHT_ELEMS (CONV1_OUT_CHANNELS * CONV1_IN_CHANNELS * CONV1_KERNEL_L * CONV1_KERNEL_L)


typedef struct network {

    _Float16 *input;
    _Float16 *output;
    short batchsize;
    conv_op conv1;

    struct {
        short conv1;
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

void metrics(_Float16 *ret, int *preds, int *labels, 
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
void backward_alexnet(alexnet *net, const int *batch_Y, const _Float16 *batch_targets, float *loss_out);

void alexnet_train(alexnet *net, int epochs);
void alexnet_test(alexnet *net);
void alexnet_inference(alexnet *net, const unsigned char *img_bytes);
void compute_batch_metrics(const int *preds, const int *labels, int batchsize);

#endif /* ALEXNET_H */
