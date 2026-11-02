//
// File:        batchnorm_layer.h
// Description: interface of the spatial batch normalization layer
//
// MiniVGGNet uses five BN layers:
//   bn1, bn2  after the block-1 convs   (C1 channels, 32x32 spatial)
//   bn3, bn4  after the block-2 convs   (C2 channels, 16x16 spatial)
//   bn5       after fc1                 (FC1_LAYER channels, 1 "spatial")
//
// During training, avg/var are recomputed from the current batch and used to
// normalise it, exactly matching PyTorch's BN train mode. Alongside, an
// exponential moving average of the batch mean/var is kept in running_mean/
// running_var (see BN_MOMENTUM). At inference (is_training == 0) the layer
// normalises with those frozen running stats instead of the batch, so a
// prediction no longer depends on the other samples in its batch and
// single-image inference is well defined.
//
#include <stdlib.h>
#include <math.h>

#define EPSILON 0.00001

// EMA weight for the running mean/var update (PyTorch's BatchNorm default):
//   running = (1 - BN_MOMENTUM) * running + BN_MOMENTUM * batch_stat
#define BN_MOMENTUM 0.1f

#ifndef ALEXNET_STATIC_MAX_BATCH
#define ALEXNET_STATIC_MAX_BATCH 32
#endif

// Architecture macros (set by the Makefile VARIANT; defaults must match alexnet.h)
#ifndef C1_CHANNELS
#define C1_CHANNELS 12
#endif
#ifndef C2_CHANNELS
#define C2_CHANNELS 24
#endif
#ifndef FC1_LAYER
#define FC1_LAYER 192
#endif

#define BN1_CHANNELS  C1_CHANNELS
#define BN2_CHANNELS  C1_CHANNELS
#define BN3_CHANNELS  C2_CHANNELS
#define BN4_CHANNELS  C2_CHANNELS
#define BN5_CHANNELS  FC1_LAYER    // 1-D BN after fc1

// Spatial sizes per BN layer
#define BN1_SPATIAL (32 * 32)
#define BN2_SPATIAL (32 * 32)
#define BN3_SPATIAL (16 * 16)
#define BN4_SPATIAL (16 * 16)
#define BN5_SPATIAL  1

#define BN1_UNITS (BN1_CHANNELS * BN1_SPATIAL)
#define BN2_UNITS (BN2_CHANNELS * BN2_SPATIAL)
#define BN3_UNITS (BN3_CHANNELS * BN3_SPATIAL)
#define BN4_UNITS (BN4_CHANNELS * BN4_SPATIAL)
#define BN5_UNITS (BN5_CHANNELS * BN5_SPATIAL)

#define BN_MAX2(a, b) ((a) > (b) ? (a) : (b))

// S1/S2 in the backward pass are stack arrays indexed by channel.
#define BN_MAX_CHANNELS \
    BN_MAX2(BN_MAX2(BN1_CHANNELS, BN3_CHANNELS), BN5_CHANNELS)

// bn_dxnorm_scratch must cover the widest BN layer (bn1/bn2: C1 x 32x32).
#define BN_MAX_UNITS \
    BN_MAX2(BN_MAX2(BN1_UNITS, BN3_UNITS), BN5_UNITS)

typedef struct batch_norm_op {
    _Float16 *input; _Float16 *d_input;
    _Float16 *output; _Float16 *d_output;
    _Float16 *gamma; _Float16 *d_gamma;
    _Float16 *beta; _Float16 *d_beta;

    int units;
    int channels;
    int spatial_size;
    short batchsize;
    short layer_id;

    _Float16 *x_norm;
    _Float16 *avg;
    _Float16 *var;

    // Running (inference-time) statistics: EMA of the per-channel batch mean
    // and unbiased variance, updated every training forward pass and used to
    // normalise when is_training == 0. Persisted in the checkpoint.
    _Float16 *running_mean;
    _Float16 *running_var;
    short  is_training;   // 1: batch stats + update running; 0: use running
} batch_norm_op;

extern _Float16 bn1_avg_buf[BN1_CHANNELS];
extern _Float16 bn2_avg_buf[BN2_CHANNELS];
extern _Float16 bn3_avg_buf[BN3_CHANNELS];
extern _Float16 bn4_avg_buf[BN4_CHANNELS];
extern _Float16 bn5_avg_buf[BN5_CHANNELS];

extern _Float16 bn1_var_buf[BN1_CHANNELS];
extern _Float16 bn2_var_buf[BN2_CHANNELS];
extern _Float16 bn3_var_buf[BN3_CHANNELS];
extern _Float16 bn4_var_buf[BN4_CHANNELS];
extern _Float16 bn5_var_buf[BN5_CHANNELS];

// Running mean/var, persisted in the checkpoint (see batchnorm_reset_running_stats).
extern _Float16 bn1_run_mean_buf[BN1_CHANNELS];
extern _Float16 bn2_run_mean_buf[BN2_CHANNELS];
extern _Float16 bn3_run_mean_buf[BN3_CHANNELS];
extern _Float16 bn4_run_mean_buf[BN4_CHANNELS];
extern _Float16 bn5_run_mean_buf[BN5_CHANNELS];

extern _Float16 bn1_run_var_buf[BN1_CHANNELS];
extern _Float16 bn2_run_var_buf[BN2_CHANNELS];
extern _Float16 bn3_run_var_buf[BN3_CHANNELS];
extern _Float16 bn4_run_var_buf[BN4_CHANNELS];
extern _Float16 bn5_run_var_buf[BN5_CHANNELS];

extern _Float16 bn1_x_norm_buf[ALEXNET_STATIC_MAX_BATCH * BN1_UNITS];
extern _Float16 bn2_x_norm_buf[ALEXNET_STATIC_MAX_BATCH * BN2_UNITS];
extern _Float16 bn3_x_norm_buf[ALEXNET_STATIC_MAX_BATCH * BN3_UNITS];
extern _Float16 bn4_x_norm_buf[ALEXNET_STATIC_MAX_BATCH * BN4_UNITS];
extern _Float16 bn5_x_norm_buf[ALEXNET_STATIC_MAX_BATCH * BN5_UNITS];

// Scratch buffer sized for the largest BN layer
extern _Float16 bn_dxnorm_scratch[ALEXNET_STATIC_MAX_BATCH * BN_MAX_UNITS];

void batch_norm_op_forward(batch_norm_op *op);
void batch_norm_op_backward(batch_norm_op *op);
void batch_norm_op_backward_full(batch_norm_op *op);
void batch_norm_op_backward_input_only(batch_norm_op *op);

void calloc_batchnorm_weights(batch_norm_op *op);
void free_batchnorm_weights(batch_norm_op *op);

void calloc_batchnorm_dweights(batch_norm_op *op);
void free_batchnorm_dweights(batch_norm_op *op);

void load_batchnorm_weights(batch_norm_op *op, const _Float16 *gamma_array, const _Float16 *beta_array);
void save_batchnorm_weights(batch_norm_op *op);

// Reset every BN layer's running stats to the identity (mean 0, var 1). Call
// before training a fresh net, or after loading a checkpoint that predates
// running stats, so inference-mode BN starts from a neutral state.
void batchnorm_reset_running_stats(void);
