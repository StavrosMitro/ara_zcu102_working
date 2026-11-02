//
// File:        activation_layer.c
// Description: Implementation of activation layer
// Author:      Haris Wang
//
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "activation_layer.h"
#include "printf.h"


typedef struct nonlinear_args {
    nonlinear_op *op;
    short batch_id;
} nonlinear_args;

static void relu_op_forward_single(void *argv)
{
    /**
     * pthread relu_op_forward
     * */
    nonlinear_args nargs;
    memcpy(&nargs, (nonlinear_args *)argv, sizeof(nonlinear_args));

    register _Float16 *input  = nargs.op->input + nargs.batch_id * (nargs.op->units);
    register _Float16 *output = nargs.op->output + nargs.batch_id * (nargs.op->units);

    // out = max(in, 0). vfmax matches the scalar `in > 0 ? in : 0` exactly:
    //   -0.0 -> scalar takes the else branch (+0.0); maxNum(-0.0, +0.0) = +0.0
    //   NaN  -> scalar takes the else branch (0);    maxNum(NaN,  +0.0) = +0.0
    // so no input value can make the two disagree.
    const _Float16 zero = 0.0f;
    int n = nargs.op->units;
    while (n > 0) {
        size_t vl;
        asm volatile("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(n));
        asm volatile("vle16.v v8, (%0)" :: "r"(input) : "memory");
        asm volatile("vfmax.vf v8, v8, %0" :: "f"(zero));
        asm volatile("vse16.v v8, (%0)" :: "r"(output) : "memory");
        input  += vl;
        output += vl;
        n -= (int)vl;
    }
}

void relu_op_forward(nonlinear_op *op)
{
    nonlinear_args args[op->batchsize+1];
    // Each sample p works on a disjoint input/output slice, so the batch loop
    // has no cross-iteration dependency and parallelises across cores.
    for(int p = 0; p < op->batchsize; p++)
    {
        args[p].op = op;
        args[p].batch_id = p;
        relu_op_forward_single((void *)(&args[p]));
    }
}

void relu_op_backward(nonlinear_op *op)
{
    // d_input[i] = (input[i] > 0) ? d_output[i] : 0
    //
    // The per-sample slices are contiguous, so this is one flat sweep over
    // batchsize*units rather than the nested loop the scalar version used.
    //
    // A masked SELECT rather than the scalar's multiply by 1.0f/0.0f: identical
    // for every finite value, and it also avoids inf*0 -> NaN on the dead side
    // (which the scalar form would produce, though -ffast-math already assumes
    // that cannot happen).
    const _Float16 zero = 0.0f;
    int n = op->batchsize * op->units;
    const _Float16 *in   = op->input;
    const _Float16 *dout = op->d_output;
    _Float16       *din  = op->d_input;

    while (n > 0) {
        size_t vl;
        asm volatile("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(n));
        asm volatile("vle16.v v8,  (%0)" :: "r"(in)   : "memory");
        asm volatile("vle16.v v16, (%0)" :: "r"(dout) : "memory");
        asm volatile("vmfgt.vf v0, v8, %0" :: "f"(zero));   // mask = input > 0
        asm volatile("vmv.v.i v24, 0");
        asm volatile("vmerge.vvm v24, v24, v16, v0");       // mask ? d_output : 0
        asm volatile("vse16.v v24, (%0)" :: "r"(din) : "memory");
        in   += vl;
        dout += vl;
        din  += vl;
        n -= (int)vl;
    }
}


#include <math.h>

// Softmax as an FP32 island: logits arrive in FP16, but max / exp / sum / divide
// are all done in float, and only the final probability is narrowed back to
// FP16. There is no FP16 transcendental, and a batch-wide exp/sum in FP16 would
// lose too much; units is small (OUT_LAYER) so a scalar pass is cheap. Kept as
// a distinct function so train.c can reuse it; the loss itself (cross_entropy)
// does its own FP32 softmax for the gradient seed.
void softmax_fc_forward_vec(const _Float16 *input, _Float16 *output, int batch_size, int units)
{
    for (int p = 0; p < batch_size; ++p) {
        const _Float16 *in_p = input + (size_t)p * units;
        _Float16       *out_p = output + (size_t)p * units;

        float max_val = (float)in_p[0];
        for (int i = 1; i < units; ++i) {
            float v = (float)in_p[i];
            if (v > max_val) max_val = v;
        }
        float esum = 0.0f;
        for (int i = 0; i < units; ++i)
            esum += expf((float)in_p[i] - max_val);
        for (int i = 0; i < units; ++i)
            out_p[i] = (_Float16)(expf((float)in_p[i] - max_val) / esum);
    }
}
