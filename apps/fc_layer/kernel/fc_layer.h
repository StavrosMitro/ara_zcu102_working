//
// File:        fc_layer.h
// Description: interface of full connected layer
// Author:      Haris Wang
//
// #include <stdlib.h>

#include <stdint.h>

#ifndef ALEXNET_STATIC_MAX_BATCH
#define ALEXNET_STATIC_MAX_BATCH 2
#endif

#define FC_MAX_IN_UNITS 2048
#define FC_MAX_INTERNAL 512

typedef struct fc_op {
    // d_input is FP16: it becomes the previous layer's FP16 d_output (the FP16
    // gradient chain). It carries the loss scale downstream (not unscaled here).
    _Float16 *input; _Float16 *d_input;
    // FP32 output / incoming-gradient — used when is_last (loss runs in FP32).
    float *output;   float *d_output;
    // FP16 chain (mirrors conv_op): forward activation + incoming gradient.
    // Primary path for hidden layers (is_last == 0); also the transient the
    // last layer downcasts its FP32 d_output into for the backward matmuls.
    _Float16 *output_f16;
    _Float16 *d_output_f16;
    _Float16 *weights; float *d_weights;   // FP32 master weights held elsewhere; op->weights is the FP16 compute copy
    float *bias;     float *d_bias;         // bias, d_weights, d_bias always FP32 (optimizer)
    int in_units, out_units;

    short batchsize;
    short layer_id;
    // AMP output-precision selector:
    //   is_last != 0 -> terminal/logits layer: FP32 output (op->output), loss in FP32
    //   is_last == 0 -> hidden layer: FP16 output (op->output_f16) via fmatmul_fused_f16out
    short is_last;
} fc_op;

typedef struct fc_backward_cycle_breakdown {
    int64_t d_input_cycles;
    int64_t d_bias_cycles;
    int64_t d_weights_cycles;
} fc_backward_cycle_breakdown;


void fc_op_forward(fc_op *op);
void fc_op_backward_full(fc_op *op);
void fc_op_backward_input_only(fc_op *op);
void fc_op_backward_full_profile(fc_op *op, fc_backward_cycle_breakdown *cycles);

inline void calloc_fc_weights(fc_op *op);
inline void free_fc_weights(fc_op *op);

inline void calloc_fc_dweights(fc_op *op);
inline void free_fc_dweights(fc_op *op);

inline void load_fc_weights(fc_op *op, float *w_array, float *b_array);
inline void save_fc_weights(fc_op *op );
