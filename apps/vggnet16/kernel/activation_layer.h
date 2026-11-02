#ifndef ACTIVATION_LAYER_H
#define ACTIVATION_LAYER_H

//
// File:        activation_layer.h
// Description: interface of activation layer
// Author:      Haris Wang
//
// #include <stdlib.h>


typedef struct nonlinear_op {
    _Float16 *input; _Float16 *d_input;
    _Float16 *output; _Float16 *d_output;
    int units;

    short batchsize;
} nonlinear_op;

void relu_op_forward(nonlinear_op *op);
void relu_op_backward(nonlinear_op *op);

void softmax_fc_forward_vec(const _Float16 *input, _Float16 *output, int batch_size, int units);

#endif // ACTIVATION_LAYER_H
