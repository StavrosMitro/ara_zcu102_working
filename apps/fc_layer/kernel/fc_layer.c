//
// File:        fc_layer.c
// Description: Implementation of full connected layer
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

#if ALEXNET_STATIC_MAX_BATCH <= 4
#define FMATMUL_MAX_M 4
#elif ALEXNET_STATIC_MAX_BATCH <= 8
#define FMATMUL_MAX_M 8
#elif ALEXNET_STATIC_MAX_BATCH <= 64
#define FMATMUL_MAX_M (((ALEXNET_STATIC_MAX_BATCH + 15) / 16) * 16)
#elif ALEXNET_STATIC_MAX_BATCH <= 128
#define FMATMUL_MAX_M (((ALEXNET_STATIC_MAX_BATCH + 7) / 8) * 8)
#else
#define FMATMUL_MAX_M (((ALEXNET_STATIC_MAX_BATCH + 3) / 4) * 4)
#endif

#define FMATMUL_MAX_N FC_MAX_IN_UNITS
#define FMATMUL_MAX_K FC_MAX_IN_UNITS

#define FC_OUTPUT_UNITS 512

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
                                         const float *bias, float *c,
                                         const int M, const int N,
                                         const int K);

static inline int64_t fc_cycle_count_local(void)
{
    int64_t cycle_count = 0;
    asm volatile("fence; csrr %0, cycle" : "=r"(cycle_count));
    return cycle_count;
}

static float fc_t_output_scratch[ALEXNET_STATIC_MAX_BATCH * FC_MAX_INTERNAL * 2];


void fc_op_forward(fc_op *op)
{
    _Float16 *fmatmul_a_scratch = (_Float16 *)shared_memory_pool;

    if (op->batchsize <= 0 || op->in_units <= 0 || op->out_units <= 0)
        return;

    unsigned long int block = fmatmul_row_block((unsigned long int)op->batchsize);
    unsigned long int padded_m = (((unsigned long int)op->batchsize + block - 1) / block) * block;

    if ((unsigned long int)op->in_units > FMATMUL_MAX_N ||
        (unsigned long int)op->out_units > FMATMUL_MAX_K ||
        padded_m > FMATMUL_MAX_M)
    {
        printf_("[SCALAR] fc_op_forward: batchsize=%d in=%d out=%d padded_m=%lu MAX_M=%d\n",
                op->batchsize, op->in_units, op->out_units, padded_m, FMATMUL_MAX_M);
        if (op->is_last) {
            // Terminal layer: FP32 logits.
            matrix_multiply_scalar_fused(op->input, op->weights, op->bias, op->output,
                                         op->batchsize, op->in_units, op->out_units);
        } else {
            // Hidden layer: compute FP32 into scratch, then narrow to FP16 output.
            matrix_multiply_scalar_fused(op->input, op->weights, op->bias, fmatmul_c_scratch,
                                         op->batchsize, op->in_units, op->out_units);
            size_t nrem = (size_t)op->batchsize * (size_t)op->out_units;
            const float *src = fmatmul_c_scratch;
            _Float16 *dst = op->output_f16;
            while (nrem > 0) {
                size_t vl;
                asm volatile("vsetvli %0, %1, e32, m8, ta, ma" : "=r"(vl) : "r"(nrem));
                asm volatile("vle32.v v24, (%0)" :: "r"(src));
                asm volatile("vsetvli zero, %0, e16, m4, ta, ma" :: "r"(vl));
                asm volatile("vfncvt.f.f.w v16, v24");
                asm volatile("vse16.v v16, (%0)" :: "r"(dst) : "memory");
                src += vl; dst += vl; nrem -= vl;
            }
        }
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

    if (op->is_last) {
        // ---- Terminal (logits) layer: FP32 output ----
        size_t mk_padded = (size_t)padded_m * (size_t)op->out_units;
        float *c_scratch_ptr = fmatmul_c_scratch;
        while(mk_padded > 0) {
            size_t vl;
            asm volatile("vsetvli %0, %1, e32, m8, ta, ma" : "=r"(vl) : "r"(mk_padded));
            asm volatile("vmv.v.i v8, 0");
            asm volatile("vse32.v v8, (%0)" :: "r"(c_scratch_ptr));
            c_scratch_ptr += vl;
            mk_padded -= vl;
        }

        fmatmul_fused(fmatmul_c_scratch, fmatmul_a_scratch, op->weights, op->bias,
                     padded_m, (unsigned long int)op->in_units,
                     (unsigned long int)op->out_units);

        size_t remaining_mk = mk;
        const float *src_mk = fmatmul_c_scratch;
        float *dst_mk = op->output;
        while (remaining_mk > 0)
        {
            size_t vl = 0;
            asm volatile("vsetvli %0, %1, e32, m1, ta, ma" : "=r"(vl) : "r"(remaining_mk));
            asm volatile("vle32.v v0, (%0);" : : "r"(src_mk) : "memory");
            asm volatile("vse32.v v0, (%0);" : : "r"(dst_mk) : "memory");
            src_mk += vl;
            dst_mk += vl;
            remaining_mk -= vl;
        }
    } else {
        // ---- Hidden layer: FP16 output via fused widening + narrow epilogue ----
        // No need to pre-zero the scratch: fmatmul_fused_f16out seeds the
        // accumulator with bias, exactly like fmatmul_fused.
        fmatmul_fused_f16out(fmatmul_c_scratch_f16, fmatmul_a_scratch, op->weights, op->bias,
                             padded_m, (unsigned long int)op->in_units,
                             (unsigned long int)op->out_units);

        size_t remaining_mk = mk;
        const _Float16 *src_mk = fmatmul_c_scratch_f16;
        _Float16 *dst_mk = op->output_f16;
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

    // Narrow the FP32 d_output (loss-scaled) into the dedicated FP16 buffer
    // d_output_f16, which feeds the FP16 backward matmuls. d_output (FP32) is kept
    // for the d_bias reduction below. (Terminal layer: op->d_output is set by the
    // loss; a hidden layer would instead arrive with op->d_output_f16 pre-filled.)
    int n_elems = op->batchsize * op->out_units;
    const float *src_grad = op->d_output;
    _Float16 *dst_grad = op->d_output_f16;

    while(n_elems > 0) {
        size_t vl;
        asm volatile("vsetvli %0, %1, e32, m8, ta, ma" : "=r"(vl) : "r"(n_elems));
        asm volatile("vle32.v v24, (%0)" :: "r"(src_grad));
        
        asm volatile("vsetvli zero, %0, e16, m4, ta, ma" :: "r"(vl));
        asm volatile("vfncvt.f.f.w v16, v24");
        asm volatile("vse16.v v16, (%0)" :: "r"(dst_grad) : "memory");
        
        src_grad += vl;
        dst_grad += vl;
        n_elems -= vl;
    }

    // d_input calculation
    t0 = fc_cycle_count_local();
    
    matrix_multiply_nt_f16out(op->d_output_f16, op->weights, op->d_input,
                              op->batchsize, op->out_units, op->in_units);

    int64_t elapsed = fc_cycle_count_local() - t0;
    if (cycles) cycles->d_input_cycles += elapsed;


    // d_bias calculation

    t0 = fc_cycle_count_local();

    calc_bias_gradient_vec(op->d_bias, op->d_output, op->out_units, op->batchsize);

    elapsed = fc_cycle_count_local() - t0;
    if (cycles) cycles->d_bias_cycles += elapsed;

    //
    //  d_weights calculation
    // 
    t0 = fc_cycle_count_local();
    register float *w_deltas = op->d_weights;

    matrix_multiply_tn(op->input, op->d_output_f16, w_deltas,
                    op->batchsize, op->in_units, op->out_units);

    elapsed = fc_cycle_count_local() - t0;
    if (cycles) cycles->d_weights_cycles += elapsed;
}


static inline void vector_scale_f32(float *vec, float scale, int length) {
    int n = length;
    while (n > 0) {
        size_t vl;

        asm volatile("vsetvli %0, %1, e32, m8, ta, ma" : "=r"(vl) : "r"(n));
        
        asm volatile("vle32.v v8, (%0)" :: "r"(vec));       
        asm volatile("vfmul.vf v8, v8, %0" :: "f"(scale));   
        asm volatile("vse32.v v8, (%0)" :: "r"(vec));

        vec += vl;
        n -= vl;
    }
}

void fc_op_backward_input_only(fc_op *op)
{
    printf_("[SCALAR] fc_op_backward_input_only: batchsize=%d in=%d out=%d\n",
            op->batchsize, op->in_units, op->out_units);
    // Only propagate gradients to previous layer when this layer is frozen.
    for (int p = 0; p < op->batchsize; p++)
    {
        for (int j = 0; j < op->out_units; j++)
        {
            register float d_o = op->d_output[p * op->out_units + j];
            for (int i = 0; i < op->in_units; i++) {
                int idx = p * op->in_units + i;
                // d_input is FP16: accumulate in FP32, store narrowed.
                op->d_input[idx] = (_Float16)((float)op->d_input[idx] +
                                              (float)op->weights[i * op->out_units + j] * d_o);
            }
        }
    }

}


void calloc_fc_weights(fc_op *op)
{
    if (op->weights)
        memset(op->weights, 0, (size_t)op->in_units * op->out_units * sizeof(_Float16));
    if (op->bias)
        memset(op->bias, 0, (size_t)op->out_units * sizeof(float));
}

void free_fc_weights(fc_op *op)
{
    (void)op;
}

void calloc_fc_dweights(fc_op *op)
{
    if (op->d_weights)
        memset(op->d_weights, 0, (size_t)op->in_units * op->out_units * sizeof(float));
    if (op->d_bias)
        memset(op->d_bias, 0, (size_t)op->out_units * sizeof(float));
}

void free_fc_dweights(fc_op *op)
{
    (void)op;
}

void save_fc_weights(fc_op *op)
{
    (void)op;
}

void load_fc_weights(fc_op *op, float *w_array, float *b_array)
{
    memcpy(op->weights, w_array, sizeof(float) * op->in_units * op->out_units);
    memcpy(op->bias, b_array, sizeof(float) * op->out_units);
}


static void matrix_multiply_scalar_fused(const _Float16 *a, const _Float16 *b,
                                         const float *bias, float *c,
                                         const int M, const int N, const int K)
{
    register int i, j, p;
    register const _Float16 *a_ptr = a;
    for (i = 0; i < M; i++)
    {
        float *c_ptr = c + i * K;
        const float *bias_ptr = bias;
        for (p = 0; p < K; p++)
            *(c_ptr++) = *(bias_ptr++);
    }
    for (i = 0; i < M; i++)
    {
        register const _Float16 *b_ptr = b;
        for (j = 0; j < N; j++)
        {
            register float apart = (float)(*(a_ptr++));
            if (apart < 0.00001f && apart > -0.00001f)
            {
                b_ptr += K;
                continue;
            }
            register float *c_ptr = c + i * K;
            for (p = 0; p < K; p++)
                *(c_ptr++) += (float)(*(b_ptr++)) * apart;
        }
    }
}

void calc_bias_gradient_vec(float *d_bias, const float *d_output, int out_units, int batchsize)
{
    float inv_batch = 1.0f / (float)batchsize;

    // d_bias is already zeroed by calloc_alexnet_d_params before backward
    for (int p = 0; p < batchsize; p++) {
        const float *row = d_output + (size_t)p * out_units;
        float *bias_ptr = d_bias;
        int j = out_units;
        while (j > 0) {
            size_t vl;
            asm volatile("vsetvli %0, %1, e32, m8, ta, ma" : "=r"(vl) : "r"(j));
            asm volatile("vle32.v v8,  (%0)" :: "r"(bias_ptr));
            asm volatile("vle32.v v16, (%0)" :: "r"(row));
            asm volatile("vfadd.vv v8, v8, v16");
            asm volatile("vse32.v v8,  (%0)" :: "r"(bias_ptr));
            bias_ptr += vl;
            row += vl;
            j -= vl;
        }
    }

    float *ptr = d_bias;
    int n = out_units;
    while (n > 0) {
        size_t vl;
        asm volatile("vsetvli %0, %1, e32, m8, ta, ma" : "=r"(vl) : "r"(n));
        asm volatile("vle32.v v8, (%0)" :: "r"(ptr));
        asm volatile("vfmul.vf v8, v8, %0" :: "f"(inv_batch));
        asm volatile("vse32.v v8, (%0)" :: "r"(ptr));
        ptr += vl;
        n -= vl;
    }
}