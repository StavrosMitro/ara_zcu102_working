//
// File:        fc_layer.c
// Description: Implementation of full connected layer
// Author:      Haris Wang
// Modified and got vectorized: Stavros Mitropoulos
//
#include <stdlib.h>
#include <math.h>
#include <string.h>
#ifdef SPIKE
#include <printf.h>
#elif defined ARA_LINUX
#include <stdio.h>
#else
#include "printf.h"
#endif
#include "fc_layer.h"
#include "matrix.h"
#include "runtime.h"
#include "fmatmul.h"

#if ALEXNET_STATIC_MAX_BATCH > 4
#define FMATMUL_MAX_M ALEXNET_STATIC_MAX_BATCH
#else
#define FMATMUL_MAX_M 4
#endif

#define FMATMUL_MAX_N FC_MAX_IN_UNITS
#define FMATMUL_MAX_K FC_MAX_IN_UNITS


static inline unsigned long int fmatmul_row_block(unsigned long int m)
{
    if (m <= 4)
        return 4;
    if (m <= 8)
        return 8;
    if (m <= 64)
        return 16;
    if (m <= 128)
        return 8;
    return 4;
}

static void matrix_multiply_scalar_fused(const _Float16 *a, const _Float16 *b,
                                         const _Float16 *bias, _Float16 *c,
                                         const int M, const int N,
                                         const int K);

static inline int64_t fc_cycle_count_local(void)
{
    int64_t cycle_count = 0;
    asm volatile("fence; csrr %0, cycle" : "=r"(cycle_count));
    return cycle_count;
}

static _Float16 fc_t_output_scratch[ALEXNET_STATIC_MAX_BATCH * FC_MAX_INTERNAL * 2];


void fc_op_forward(fc_op *op)
{
    _Float16 *fmatmul_a_scratch = shared_memory_pool;

    if (op->batchsize <= 0 || op->in_units <= 0 || op->out_units <= 0)
        return;

    unsigned long int block = fmatmul_row_block((unsigned long int)op->batchsize);
    unsigned long int padded_m = (((unsigned long int)op->batchsize + block - 1) / block) * block;

    if ((unsigned long int)op->in_units > FMATMUL_MAX_N ||
        (unsigned long int)op->out_units > FMATMUL_MAX_K ||
        padded_m > FMATMUL_MAX_M)
    {
        matrix_multiply_scalar_fused(op->input, op->weights, op->bias, op->output,
                                     op->batchsize, op->in_units, op->out_units);
        return;
    }

    const size_t mn = (size_t)op->batchsize * (size_t)op->in_units;
    const size_t pnk = (size_t)padded_m * (size_t)op->in_units;
    const size_t mk = (size_t)op->batchsize * (size_t)op->out_units;

    size_t remaining_mn = mn;
    const _Float16 *src_mn = op->input;
    _Float16 *dst_mn = fmatmul_a_scratch;
    while (remaining_mn > 0)
    {
        size_t vl = 0;
        asm volatile("vsetvli %0, %1, e16, m1, ta, ma" : "=r"(vl) : "r"(remaining_mn));
        asm volatile("vle16.v v0, (%0);" : : "r"(src_mn) : "memory");
        asm volatile("vse16.v v0, (%0);" : : "r"(dst_mn) : "memory");
        src_mn += vl;
        dst_mn += vl;
        remaining_mn -= vl;
    }

    size_t remaining = pnk - mn;
    _Float16 *dst = fmatmul_a_scratch + mn;
    while (remaining > 0)
    {
        size_t vl = 0;
        asm volatile("vsetvli %0, %1, e16, m1, ta, ma" : "=r"(vl) : "r"(remaining));
        asm volatile("vmv.v.x v0, zero");
        asm volatile("vse16.v v0, (%0);" : : "r"(dst) : "memory");
        dst += vl;
        remaining -= vl;
    }

    fmatmul_fused(fmatmul_c_scratch, fmatmul_a_scratch, op->weights, op->bias,
                 padded_m, (unsigned long int)op->in_units,
                 (unsigned long int)op->out_units);

    size_t remaining_mk = mk;
    const _Float16 *src_mk = fmatmul_c_scratch;
    _Float16 *dst_mk = op->output;
    while (remaining_mk > 0)
    {
        size_t vl = 0;
        asm volatile("vsetvli %0, %1, e16, m1, ta, ma" : "=r"(vl) : "r"(remaining_mk));
        asm volatile("vle16.v v0, (%0);" : : "r"(src_mk) : "memory");
        asm volatile("vse16.v v0, (%0);" : : "r"(dst_mk) : "memory");
        src_mk += vl;
        dst_mk += vl;
        remaining_mk -= vl;
    }
}


void fc_op_backward_full_profile(fc_op *op, fc_backward_cycle_breakdown *cycles)
{
    int64_t t0 = 0;

    if (cycles) {
        cycles->d_input_cycles = 0;
        cycles->d_bias_cycles = 0;
        cycles->d_weights_cycles = 0;
    }

    if (op->d_weights == NULL || op->d_bias == NULL) {
        fc_op_backward_input_only(op);
        return;
    }

    // calculate delta_input per sample using A * B^T
    t0 = fc_cycle_count_local();

    // d_input calculation
    matrix_multiply_nt_deferred(op->d_output, op->weights, op->d_input,
                       op->batchsize, op->out_units, op->in_units); 
    


    int64_t elapsed = fc_cycle_count_local() - t0;
    if (cycles)
        cycles->d_input_cycles += elapsed;

    // calculate delta_bias averaged across batch
    t0 = fc_cycle_count_local();

    calc_bias_gradient_vec_batch2(op->d_bias, op->d_output, op->out_units);

    elapsed = fc_cycle_count_local() - t0;
    if (cycles)
        cycles->d_bias_cycles += elapsed;

    t0 = fc_cycle_count_local();
    // calculate delta_weights
    register _Float16 *w_deltas = op->d_weights;

    matrix_multiply_tn(op->input, op->d_output, w_deltas,
                    op->batchsize, op->in_units, op->out_units);


    elapsed = fc_cycle_count_local() - t0;
    if (cycles)
        cycles->d_weights_cycles += elapsed;
}

static inline void vector_scale_f16(_Float16 *vec, _Float16 scale, int length) {
    int n = length;
    while (n > 0) {
        size_t vl;

        asm volatile("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(n));
        
        asm volatile("vle16.v v8, (%0)" :: "r"(vec));        // Φόρτωση
        asm volatile("vfmul.vf v8, v8, %0" :: "f"(scale));   // Πολλαπλασιασμός
        asm volatile("vse16.v v8, (%0)" :: "r"(vec));        // Εγγραφή

        vec += vl;
        n -= vl;
    }
}

void fc_op_backward_input_only(fc_op *op)
{
    // Only propagate gradients to previous layer when this layer is frozen.
    for (int p = 0; p < op->batchsize; p++)
    {
        for (int j = 0; j < op->out_units; j++)
        {
            register _Float16 d_o = op->d_output[p * op->out_units + j];
            for (int i = 0; i < op->in_units; i++)
                op->d_input[p * op->in_units + i] += op->weights[i * op->out_units + j] * d_o;
        }
    }

}


void calloc_fc_weights(fc_op *op)
{
    if (op->weights)
        memset(op->weights, 0, (size_t)op->in_units * op->out_units * sizeof(_Float16));
    if (op->bias)
        memset(op->bias, 0, (size_t)op->out_units * sizeof(_Float16));
}

void free_fc_weights(fc_op *op)
{
    (void)op;
}

void calloc_fc_dweights(fc_op *op)
{
    if (op->d_weights)
        memset(op->d_weights, 0, (size_t)op->in_units * op->out_units * sizeof(_Float16));
    if (op->d_bias)
        memset(op->d_bias, 0, (size_t)op->out_units * sizeof(_Float16));
}

void free_fc_dweights(fc_op *op)
{
    (void)op;
}

void save_fc_weights(fc_op *op)
{
    (void)op;
}

void load_fc_weights(fc_op *op, _Float16 *w_array, _Float16 *b_array)
{
    memcpy(op->weights, w_array, sizeof(_Float16) * op->in_units * op->out_units);
    memcpy(op->bias, b_array, sizeof(_Float16) * op->out_units);
}


static void matrix_multiply_scalar_fused(const _Float16 *a, const _Float16 *b,
                                         const _Float16 *bias, _Float16 *c,
                                         const int M, const int N, const int K)
{
    register int i, j, p;
    register const _Float16 *a_ptr = a;
    for (i = 0; i < M; i++)
    {
        _Float16 *c_ptr = c + i * K;
        const _Float16 *bias_ptr = bias;
        for (p = 0; p < K; p++)
            *(c_ptr++) = *(bias_ptr++);
    }
    for (i = 0; i < M; i++)
    {
        register const _Float16 *b_ptr = b;
        for (j = 0; j < N; j++)
        {
            register _Float16 apart = *(a_ptr++);
            if (apart < 0.00001f && apart > -0.00001f)
            {
                b_ptr += K;
                continue;
            }
            register _Float16 *c_ptr = c + i * K;
            for (p = 0; p < K; p++)
                *(c_ptr++) += *(b_ptr++) * apart;
        }
    }
}

void calc_bias_gradient_vec_batch2(_Float16 *d_bias, const _Float16 *d_output, int out_units) 
{
    /*
    // calculate delta_bias averaged across batch
    // Initialize bias accumulator to zero
    for (int j = 0; j < op->out_units; j++)
        op->d_bias[j] = 0.0f;

    // Accumulate over batch, row by row
    for (int p = 0; p < op->batchsize; p++) {
        _Float16* row = &(op->d_output[p * op->out_units]); //start of a batch
        for (int j = 0; j < op->out_units; j++) {
            op->d_bias[j] += row[j];
        }
    }
    */
    // Explicit pointers  (batchsize = 2)
    const _Float16 *row0 = d_output;
    const _Float16 *row1 = d_output + out_units;
    _Float16 *bias_ptr = d_bias;

    // DIV 2
    _Float16 inv_batch = 0.5f; 

    int j = out_units;
    
    while (j > 0) {
        size_t vl;
        
        //LMUL=8 
        asm volatile("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(j));

        asm volatile("vle16.v v8, (%0)" :: "r"(row0));

        asm volatile("vle16.v v16, (%0)" :: "r"(row1));

        asm volatile("vfadd.vv v8, v8, v16");

        asm volatile("vfmul.vf v8, v8, %0" :: "f"(inv_batch));

        asm volatile("vse16.v v8, (%0)" :: "r"(bias_ptr));

        row0 += vl;
        row1 += vl;
        bias_ptr += vl;
        j -= vl;
    }
}