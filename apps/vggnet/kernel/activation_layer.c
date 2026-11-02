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

    register float *input  = nargs.op->input + nargs.batch_id * (nargs.op->units);
    register float *output = nargs.op->output + nargs.batch_id * (nargs.op->units);

    // out = max(in, 0). vfmax matches the scalar `in > 0 ? in : 0` exactly:
    //   -0.0 -> scalar takes the else branch (+0.0); maxNum(-0.0, +0.0) = +0.0
    //   NaN  -> scalar takes the else branch (0);    maxNum(NaN,  +0.0) = +0.0
    // so no input value can make the two disagree.
    const float zero = 0.0f;
    int n = nargs.op->units;
    while (n > 0) {
        size_t vl;
        asm volatile("vsetvli %0, %1, e32, m8, ta, ma" : "=r"(vl) : "r"(n));
        asm volatile("vle32.v v8, (%0)" :: "r"(input) : "memory");
        asm volatile("vfmax.vf v8, v8, %0" :: "f"(zero));
        asm volatile("vse32.v v8, (%0)" :: "r"(output) : "memory");
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
    const float zero = 0.0f;
    int n = op->batchsize * op->units;
    const float *in   = op->input;
    const float *dout = op->d_output;
    float       *din  = op->d_input;

    while (n > 0) {
        size_t vl;
        asm volatile("vsetvli %0, %1, e32, m8, ta, ma" : "=r"(vl) : "r"(n));
        asm volatile("vle32.v v8,  (%0)" :: "r"(in)   : "memory");
        asm volatile("vle32.v v16, (%0)" :: "r"(dout) : "memory");
        asm volatile("vmfgt.vf v0, v8, %0" :: "f"(zero));   // mask = input > 0
        asm volatile("vmv.v.i v24, 0");
        asm volatile("vmerge.vvm v24, v24, v16, v0");       // mask ? d_output : 0
        asm volatile("vse32.v v24, (%0)" :: "r"(din) : "memory");
        in   += vl;
        dout += vl;
        din  += vl;
        n -= (int)vl;
    }
}


#include <float.h>
#include <riscv_vector.h>
// __exp_2xf32: Cephes polynomial exp, shared with apps/exp and apps/softmax.
// Reached via -I$(CURDIR)/exp/kernel on the vggnet build.
#include "exp.h"

float find_max_rvv(const float *in_p, int units) {
    int avl = units;
    size_t vl = __riscv_vsetvl_e32m1(avl); 
    vfloat32m1_t v_max = __riscv_vfmv_v_f_f32m1(-FLT_MAX, vl);  
    int idx = 0;
    while (avl > 0) {
        vl = __riscv_vsetvl_e32m1(avl);
        vfloat32m1_t v_in = __riscv_vle32_v_f32m1(in_p + idx, vl);
        

        v_max = __riscv_vfredmax_vs_f32m1_f32m1(v_in, v_max, vl);
        
        idx += vl;
        avl -= vl;
    }
    float m = __riscv_vfmv_f_s_f32m1_f32(v_max);
    
    return m;
}

void softmax_fc_forward_vec(const float *input, float *output, int batch_size, int units) 
{
    for (int p = 0; p < batch_size; ++p) {
        const float *in_p = input + p * units;
        float *out_p = output + p * units;
        
        int avl = units;
        size_t vl;
        
        float max_val = find_max_rvv(in_p, units);

        float esum = 0.0f;
        avl = units;
        int idx = 0;
        
        while (avl > 0) {
            vl = __riscv_vsetvl_e32m1(avl);
            
            vfloat32m1_t v_in = __riscv_vle32_v_f32m1(in_p + idx, vl);
            vfloat32m1_t v_sub = __riscv_vfsub_vf_f32m1(v_in, max_val, vl);
            
            vfloat32m1_t v_exp = __exp_2xf32(v_sub, vl); 
            
            __riscv_vse32_v_f32m1(out_p + idx, v_exp, vl);
            
            
            idx += vl;
            avl -= vl;
        }

        for (int i = 0; i < units; ++i) {
            esum += out_p[i];
        }

        avl = units;
        idx = 0;
        while (avl > 0) {
            vl = __riscv_vsetvl_e32m1(avl);
            vfloat32m1_t v_exp = __riscv_vle32_v_f32m1(out_p + idx, vl);
            vfloat32m1_t v_res = __riscv_vfdiv_vf_f32m1(v_exp, esum, vl);
            __riscv_vse32_v_f32m1(out_p + idx, v_res, vl);
            idx += vl;
            avl -= vl;
        }
    }
}