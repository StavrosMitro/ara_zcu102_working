#ifndef ACTIVATION_LAYER_H
#define ACTIVATION_LAYER_H

//
// File:        activation_layer.h
// Description: interface of activation layer
// Author:      Haris Wang
//
// #include <stdlib.h>


typedef struct nonlinear_op {
    float *input; float *d_input;
    float *output; float *d_output;
    int units;

    short batchsize;
} nonlinear_op;

void relu_op_forward(nonlinear_op *op);
void relu_op_backward(nonlinear_op *op);

void softmax_fc_forward_vec(const float *input, float *output, int batch_size, int units);

#endif // ACTIVATION_LAYER_H
