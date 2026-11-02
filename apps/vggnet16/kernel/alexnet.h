//
// File:        alexnet.h
// Description: MiniVGGNet definition for CIFAR-100
//
// Architecture (32x32x3 input), following the reference MiniVGGNet table.
// Note the ordering inside each block is CONV -> ACT -> BN, and there is
// no dropout anywhere in the network.
//
//   INPUT    3  x 32 x 32
//   CONV1   C1  x 32 x 32   3x3, stride 1, pad 1
//   ACT     relu
//   BN1                     spatial BN over C1 channels
//   CONV2   C1  x 32 x 32   3x3, stride 1, pad 1
//   ACT     relu
//   BN2
//   POOL1   C1  x 16 x 16   2x2, stride 2
//   CONV3   C2  x 16 x 16   3x3, stride 1, pad 1
//   ACT     relu
//   BN3
//   CONV4   C2  x 16 x 16   3x3, stride 1, pad 1
//   ACT     relu
//   BN4
//   POOL2   C2  x  8 x  8   2x2, stride 2
//   FLATTEN C2 * 8 * 8
//   FC1     FC1_LAYER
//   ACT     relu
//   BN5                     1-D BN over FC1_LAYER units
//   FC2     OUT_LAYER       logits
//   SOFTMAX                 fused into the cross-entropy loss (see train.c)
//
// Widths are set by a uniform multiplier alpha applied to the reference
// 32/32/64/64/512 configuration. The Makefile VARIANT selects alpha:
//   full  alpha=1.0    32 / 64 / 512   -> 2,215,940 params
//   alpha alpha=0.375  12 / 24 / 192   ->   324,400 params  (default)
//
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
#include "convolution_layer.h"
#include "maxpooling_layer.h"
#include "activation_layer.h"
#include "fc_layer.h"
#include "batchnorm_layer.h"
#include "dropout_layer.h"

//#define SHOW_PREDCITION_DETAIL
//#define SHOW_METRIC_EVALUTE
//#define SHOW_OP_TIME

#define IN_CHANNELS  3

// Channel counts -- overridable via -DC1_CHANNELS=32 etc. (VARIANT in Makefile)
#ifndef C1_CHANNELS
#define C1_CHANNELS  12
#endif
#ifndef C2_CHANNELS
#define C2_CHANNELS  24
#endif
#ifndef FC1_LAYER
#define FC1_LAYER   192
#endif
#ifndef OUT_LAYER
#define OUT_LAYER   100   // CIFAR-100; pass -DOUT_LAYER=10 for CIFAR-10
#endif

// Dropout keep-probabilities per the reference MiniVGGNet table: after each
// pooling block, and after the FC head. Set any to 0 to disable that dropout
// (e.g. -DDROPOUT1=0 -DDROPOUT2=0 -DDROPOUT_FC=0 reproduces the no-dropout net).
#ifndef DROPOUT1
#define DROPOUT1    0.25f   // after pool1
#endif
#ifndef DROPOUT2
#define DROPOUT2    0.25f   // after pool2
#endif
#ifndef DROPOUT_FC
#define DROPOUT_FC  0.50f   // after the FC head (bn5), before fc2
#endif

// Every conv is 3x3 / stride 1 / pad 1, so spatial size is preserved.
#define CONV_KERNEL_L 3
#define CONV_STRIDES  1
#define CONV_PADDING  1

// Every pool is 2x2 / stride 2, so spatial size halves.
#define POOL_KERNEL_L 2
#define POOL_STRIDES  2

// Spatial dimensions through the network
#define FEATURE0_L  32   // input
#define FEATURE1_L  32   // block 1 convs
#define POOLING1_L  16   // after pool1
#define FEATURE2_L  16   // block 2 convs
#define POOLING2_L   8   // after pool2

// Flattened pool2 output -- one input sample to fc1
#define FC1_IN_UNITS (C2_CHANNELS * POOLING2_L * POOLING2_L)

typedef struct network {

    _Float16 *input;
    _Float16 *output;
    short batchsize;
    short is_training;

    // Block 1: conv1 -> relu1 -> bn1 -> conv2 -> relu2 -> bn2 -> pool1
    conv_op        conv1;
    nonlinear_op   relu1;
    batch_norm_op  bn1;
    conv_op        conv2;
    nonlinear_op   relu2;
    batch_norm_op  bn2;
    max_pooling_op pool1;

    // Block 2: conv3 -> relu3 -> bn3 -> conv4 -> relu4 -> bn4 -> pool2
    conv_op        conv3;
    nonlinear_op   relu3;
    batch_norm_op  bn3;
    conv_op        conv4;
    nonlinear_op   relu4;
    batch_norm_op  bn4;
    max_pooling_op pool2;

    // Classifier: fc1 -> relu5 -> bn5 -> fc2
    fc_op          fc1;
    nonlinear_op   relu5;
    batch_norm_op  bn5;   // 1-D BN over FC1_LAYER units
    fc_op          fc2;

    struct {
        short conv1, conv2, conv3, conv4;
        short bn1, bn2, bn3, bn4, bn5;
        short fc1, fc2;
    } trainable;
} alexnet;


//
//  Metric type constants
//
#define METRIC_ACCURACY  0
#define METRIC_PRECISION 1
#define METRIC_RECALL    2
#define METRIC_F1SCORE   3
#define METRIC_ROC       4

void metrics(float *ret, int *preds, int *labels,
                int classes, int TotalNum, int type);
int argmax(_Float16 *arr, int n);

void malloc_alexnet(alexnet *net);
void free_alexnet(alexnet *net);
void alexnet_set_all_trainable(alexnet *net, short trainable);
// FINETUNE_MODE: freeze the conv stack, leave only the classifier trainable.
void alexnet_set_finetune_trainable(alexnet *net);
int  alexnet_param_count(void);

void setup_alexnet(alexnet *net, short batchsize);
void alexnet_init_weights(alexnet *net);

void forward_alexnet(alexnet *net);
void free_forward_activations(alexnet *net);
void save_alexnet(alexnet *net);
// Returns 1 on success. Fails if the file's size does not match this build's
// parameter count, which catches an architecture/checkpoint mismatch.
int  load_alexnet_from_file(alexnet *net, const char *path);
// Full backward pass (conv stack + classifier) and weight update.
// Returns mean cross-entropy loss over the batch.
float backward_alexnet(alexnet *net, int *batch_Y);

void alexnet_train(alexnet *net, int epochs);
void alexnet_test(alexnet *net);
void alexnet_inference(alexnet *net, const unsigned char *img_bytes);
void compute_batch_metrics(const int *preds, const int *labels, int batchsize);
