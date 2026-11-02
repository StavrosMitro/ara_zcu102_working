//
// fp16pure/fc_layer_p16.h
// Pure FP16 fc_op interface with _p16-suffixed names so both this and the
// mixed-precision fp16/fc_layer.h can coexist in the same test binary.
//
#ifndef FC_LAYER_P16_H
#define FC_LAYER_P16_H

#include <stdint.h>

#ifndef ALEXNET_STATIC_MAX_BATCH
#define ALEXNET_STATIC_MAX_BATCH 2
#endif

// Mirror of fc_op (fc_layer16only) with all-FP16 data pointers.
typedef struct fc_op_p16 {
    _Float16 *input;   _Float16 *d_input;
    _Float16 *output;  _Float16 *d_output;
    _Float16 *weights; _Float16 *d_weights;
    _Float16 *bias;    _Float16 *d_bias;
    int in_units, out_units;
    short batchsize;
    short layer_id;
} fc_op_p16;

void fc_op_forward_p16(fc_op_p16 *op);

#endif // FC_LAYER_P16_H
