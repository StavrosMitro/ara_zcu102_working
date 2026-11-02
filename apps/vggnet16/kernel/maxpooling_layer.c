//
// File:        maxpooling_layer.c
// Description: Implementation of max pooling layer
// Author:      Haris Wang
//
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdint.h>
#include "maxpooling_layer.h"
#include "utils.h"
#ifdef SPIKE
#include <printf.h>
#elif defined ARA_LINUX
#include <stdio.h>
#else
#include "printf.h"
#endif

#define MAX(a,b) (((a) > (b)) ? (a) : (b))
#define MIN(a,b) (((a) < (b)) ? (a) : (b))


typedef struct mp_args{
    max_pooling_op *op;
    short batch_id;
} mp_args;

static void mp_op_forward_single(void *argv)
{
    // ONLY FOR 2X2 MAX POOLING
    mp_args mp = *(const mp_args *)argv;
    register _Float16 *input  = mp.op->input + mp.batch_id * mp.op->in_units;
    register _Float16 *output = mp.op->output + mp.batch_id * mp.op->out_units;
    int channels  = mp.op->channels;
    int strides   = mp.op->stride;
    int pool_size = mp.op->kernel_size;

    register int o_x, o_y;
    register int input_offset;
    register int output_offset;
    register int iwih = mp.op->in_w * mp.op->in_h;
    register int owoh = mp.op->out_w * mp.op->out_h;
    register int16_t *indices = mp.op->max_indices + mp.batch_id * mp.op->out_units;

    for (register int c = 0; c < channels; c++)
    {
        for (int j = 0; j < mp.op->in_h-strides+1; j += strides) {

            int o_w_left = mp.op->out_w; 

            int i = 0;                   
            int o_x = 0;     
            register unsigned long byte_stride = 2 * sizeof(_Float16);            

            while (o_w_left > 0) {
                unsigned long int vl;
                asm volatile("vsetvli %0, %1, e16, m4, ta, ma" : "=r"(vl) : "r"(o_w_left));

                _Float16 *row0 = &input[i + j * mp.op->in_w + c * iwih];
                _Float16 *row1 = row0 + mp.op->in_w;
                int16_t *idx_ptr = &indices[o_x + (j/2) * mp.op->out_w + c * owoh]; 
                _Float16 *out_ptr = &output[o_x + (j/2) * mp.op->out_w + c * owoh];

                asm volatile("vlse16.v v4,  (%0), %1" : : "r"(row0),     "r"(byte_stride)); // P0
                asm volatile("vlse16.v v8,  (%0), %1" : : "r"(row0 + 1), "r"(byte_stride)); // P1

                asm volatile("vlse16.v v12, (%0), %1" : : "r"(row1),     "r"(byte_stride)); // P2
                asm volatile("vlse16.v v16, (%0), %1" : : "r"(row1 + 1), "r"(byte_stride)); // P3

                // P0..P3 must all stay live in v4/v8/v12/v16, because the argmax
                // chain below compares against every one of them. So the max tree
                // accumulates into v24/v28 rather than overwriting an operand.
                // (All group bases stay 4-aligned, as LMUL=4 requires.)
                asm volatile("vfmax.vv v24, v4,  v8");
                asm volatile("vfmax.vv v24, v24, v12");
                asm volatile("vfmax.vv v28, v24, v16");                                     // overall max
                asm volatile("vse16.v v28, (%0)" : : "r"(out_ptr));

                // Tie-break MUST match the scalar reference. That version scans the
                // window x-outer / y-inner with a strict '>', so the FIRST maximum
                // wins, giving priority P0 > P2 > P1 > P3. vmerge keeps the LAST
                // match, so the merges are applied lowest-priority-first and P0
                // overwrites last. Getting this backwards is not a correctness bug
                // -- max has no unique subgradient at a tie -- but it makes the
                // vectorized build stop reproducing the scalar reference, which is
                // the comparison used to validate every other kernel here.
                asm volatile(
                    "vmv.v.i v20, 3 \n\t"               // default: P3, lowest priority
                    "vmfeq.vv v0, v28, v8 \n\t"
                    "vmerge.vim v20, v20, 1, v0 \n\t"   // P1
                    "vmfeq.vv v0, v28, v12 \n\t"
                    "vmerge.vim v20, v20, 2, v0 \n\t"   // P2
                    "vmfeq.vv v0, v28, v4 \n\t"
                    "vmerge.vim v20, v20, 0, v0"        // P0, highest priority
                );
                asm volatile("vse16.v v20, (%0)" : : "r"(idx_ptr));

                i += vl * 2; // Stride 2
                o_x += vl;
                o_w_left -= vl;
            }
        }
    }
}

void max_pooling_op_forward(max_pooling_op *op)
{
    mp_args args[op->batchsize+1];
    for (int p = 0; p < op->batchsize; p++)
    {
        args[p].op = op;
        args[p].batch_id = p;
        mp_op_forward_single((void *)(&args[p]));
    }
}

static void mp_op_backward_single(void *argv)
{
    // ONLY FOR 2X2 MAX POOLING
    mp_args mp = *(const mp_args *)argv;
    
    register _Float16 *d_input  = mp.op->d_input + mp.batch_id * mp.op->in_units;
    register _Float16 *d_output = mp.op->d_output + mp.batch_id * mp.op->out_units;
    register int16_t *indices  = mp.op->max_indices + mp.batch_id * mp.op->out_units;

    int channels  = mp.op->channels;
    int strides   = mp.op->stride;
    int in_w      = mp.op->in_w;
    int in_h      = mp.op->in_h;
    int out_w     = mp.op->out_w;
    
    register int iwih = in_w * in_h;
    register int owoh = out_w * mp.op->out_h;


    memset_vectorized_zero_f32(d_input, (size_t)channels * (size_t)iwih);

    for (register int c = 0; c < channels; c++)
    {
        for (int j = 0; j < in_h - strides + 1; j += strides) {

            int o_w_left = out_w; 
            int i = 0;                   
            int o_x = 0;     
            register unsigned long byte_stride = 2 * sizeof(_Float16);            

            while (o_w_left > 0) {
                unsigned long int vl;
                asm volatile("vsetvli %0, %1, e16, m4, ta, ma" : "=r"(vl) : "r"(o_w_left));

                _Float16 *row0 = &d_input[i + j * in_w + c * iwih];
                _Float16 *row1 = row0 + in_w;
                
                _Float16 *d_out_ptr = &d_output[o_x + (j/2) * out_w + c * owoh];
                int16_t *idx_ptr   = &indices[o_x + (j/2) * out_w + c * owoh];

                asm volatile(
                    "vle16.v v28, (%0) \n\t"        
                    "vle16.v v20, (%1) \n\t"
                    
                    
                    "vmseq.vi v0, v20, 0 \n\t"
                    "vsse16.v v28, (%2), %6, v0.t \n\t"
                    
                    "vmseq.vi v0, v20, 1 \n\t"
                    "vsse16.v v28, (%3), %6, v0.t \n\t"
                    
                    "vmseq.vi v0, v20, 2 \n\t"
                    "vsse16.v v28, (%4), %6, v0.t \n\t"
                    
                    "vmseq.vi v0, v20, 3 \n\t"
                    "vsse16.v v28, (%5), %6, v0.t"
                    :
                    : "r"(d_out_ptr), "r"(idx_ptr),
                      "r"(row0), "r"(row0 + 1), "r"(row1), "r"(row1 + 1),
                      "r"(byte_stride)
                );

                i += vl * 2; 
                o_x += vl;
                o_w_left -= vl;
            }
        }
    }
}

#ifdef DUMP_TRACE
// One-shot audit: recompute the argmax with the SCALAR algorithm (x outer,
// y inner, strict '>', first wins) and compare against the code the vectorized
// forward stored in max_indices. Any mismatch means the two implementations
// disagree about which input receives the gradient.
static void mp_audit_indices(max_pooling_op *op)
{
    static int audited = 0;
    if (audited >= 2) return;     // audit both pools, not just the first call
    audited++;

    const int ch = op->channels, in_w = op->in_w, in_h = op->in_h;
    const int out_w = op->out_w, out_h = op->out_h, ps = op->kernel_size;
    const int iwih = in_w * in_h, owoh = out_w * out_h;
    long mismatch = 0, ties = 0, total = 0, valmis = 0;
    int first_p = -1, first_c = -1, first_j = -1, first_i = -1, first_s = -1, first_v = -1;

    for (int p = 0; p < op->batchsize; p++)
    for (int c = 0; c < ch; c++)
    for (int j = 0; j < out_h; j++)
    for (int i = 0; i < out_w; i++) {
        int x0 = i * ps, y0 = j * ps;
        _Float16 best = -1111111.0f; int bx = 0, by = 0, neq = 0;
        for (int x = x0; x < MIN(x0 + ps, in_w); x++)
            for (int y = y0; y < MIN(y0 + ps, in_h); y++) {
                _Float16 v = op->input[p*ch*iwih + c*iwih + y*in_w + x];
                if (v > best) { best = v; bx = x; by = y; neq = 1; }
                else if (v == best) neq++;
            }
        int scalar_code = (by - y0) * 2 + (bx - x0);
        int oidx = p*ch*owoh + c*owoh + j*out_w + i;
        int vec_code = op->max_indices[oidx];
        // The scalar backward would have done d_input[target] += d_output[oidx];
        // starting from a zeroed buffer, so d_input[target] must equal d_output.
        int target = p*ch*iwih + c*iwih + by*in_w + bx;
        if (op->d_input[target] != op->d_output[oidx]) valmis++;
        total++;
        if (neq > 1) ties++;
        if (scalar_code != vec_code) {
            if (mismatch == 0) {
                first_p = p; first_c = c; first_j = j; first_i = i;
                first_s = scalar_code; first_v = vec_code;
            }
            mismatch++;
        }
    }
    printf_("[MPAUDIT] units=%ld  ties=%ld  index_mismatch=%ld  value_mismatch=%ld", total, ties, mismatch, valmis);
    if (mismatch)
        printf_("  first at p=%d c=%d row=%d col=%d scalar=%d vec=%d",
                first_p, first_c, first_j, first_i, first_s, first_v);
    printf_("\n");
}
#endif

void max_pooling_op_backward(max_pooling_op *op)
{
    mp_args args[op->batchsize+1];
    for (int p = 0; p < op->batchsize; p++)
    {
        args[p].op = op;
        args[p].batch_id = p;
        mp_op_backward_single((void *)(&args[p]));
    }
#ifdef DUMP_TRACE
    mp_audit_indices(op);
#endif
}
