//
// File:        fc_layer.h
// Description: interface of full connected layer
// Author:      Haris Wang
//
#ifndef FC_LAYER_H
#define FC_LAYER_H
// #include <stdlib.h>

#include <stdint.h>

#ifndef ALEXNET_STATIC_MAX_BATCH
#define ALEXNET_STATIC_MAX_BATCH 2
#endif

#define FC_MAX_IN_UNITS 2048
#define FC_MAX_INTERNAL 512

typedef struct fc_op {
    float *input;   float *d_input;
    float *output;  float *d_output;
    float *weights; float *d_weights;
    float *bias;    float *d_bias;
    int in_units, out_units;

    short batchsize;
    short layer_id;
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

#endif /* FC_LAYER_H */
