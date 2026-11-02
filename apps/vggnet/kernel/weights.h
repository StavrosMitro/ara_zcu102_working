#ifndef WEIGHTS_H
#define WEIGHTS_H

// This header must be included after alexnet.h so that C1_CHANNELS, C2_CHANNELS,
// FC1_LAYER, FC1_IN_UNITS, OUT_LAYER, IN_CHANNELS and CONV_KERNEL_L are defined.
//
// These arrays are the network's parameter storage. They live in .bss and are
// filled either by alexnet_init_weights() (training from scratch) or by
// load_alexnet_from_file() (fine-tuning a checkpoint written by save_alexnet).

#define CONV_KK (CONV_KERNEL_L * CONV_KERNEL_L)

extern float conv1_weights[C1_CHANNELS * IN_CHANNELS * CONV_KK];
extern float conv1_bias   [C1_CHANNELS];
extern float conv2_weights[C1_CHANNELS * C1_CHANNELS * CONV_KK];
extern float conv2_bias   [C1_CHANNELS];
extern float conv3_weights[C2_CHANNELS * C1_CHANNELS * CONV_KK];
extern float conv3_bias   [C2_CHANNELS];
extern float conv4_weights[C2_CHANNELS * C2_CHANNELS * CONV_KK];
extern float conv4_bias   [C2_CHANNELS];

extern float fc1_weights[FC1_IN_UNITS * FC1_LAYER];
extern float fc1_bias   [FC1_LAYER];
extern float fc2_weights[FC1_LAYER * OUT_LAYER];
extern float fc2_bias   [OUT_LAYER];

extern float bn1_gamma[C1_CHANNELS]; extern float bn1_beta[C1_CHANNELS];
extern float bn2_gamma[C1_CHANNELS]; extern float bn2_beta[C1_CHANNELS];
extern float bn3_gamma[C2_CHANNELS]; extern float bn3_beta[C2_CHANNELS];
extern float bn4_gamma[C2_CHANNELS]; extern float bn4_beta[C2_CHANNELS];
extern float bn5_gamma[FC1_LAYER];   extern float bn5_beta[FC1_LAYER];

#endif
