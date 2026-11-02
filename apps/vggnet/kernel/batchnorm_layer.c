//
// File:        batchnorm_layer.c
// Description: Implementation of batch normalization layer
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
#include "batchnorm_layer.h"
#include "utils.h"

float bn1_avg_buf[BN1_CHANNELS];
float bn2_avg_buf[BN2_CHANNELS];
float bn3_avg_buf[BN3_CHANNELS];
float bn4_avg_buf[BN4_CHANNELS];
float bn5_avg_buf[BN5_CHANNELS];

float bn1_var_buf[BN1_CHANNELS];
float bn2_var_buf[BN2_CHANNELS];
float bn3_var_buf[BN3_CHANNELS];
float bn4_var_buf[BN4_CHANNELS];
float bn5_var_buf[BN5_CHANNELS];

// Running (inference-time) statistics. Initialised to mean 0 / var 1 by
// batchnorm_reset_running_stats(); updated as an EMA during training.
float bn1_run_mean_buf[BN1_CHANNELS];
float bn2_run_mean_buf[BN2_CHANNELS];
float bn3_run_mean_buf[BN3_CHANNELS];
float bn4_run_mean_buf[BN4_CHANNELS];
float bn5_run_mean_buf[BN5_CHANNELS];

float bn1_run_var_buf[BN1_CHANNELS];
float bn2_run_var_buf[BN2_CHANNELS];
float bn3_run_var_buf[BN3_CHANNELS];
float bn4_run_var_buf[BN4_CHANNELS];
float bn5_run_var_buf[BN5_CHANNELS];

float bn1_x_norm_buf[ALEXNET_STATIC_MAX_BATCH * BN1_UNITS];
float bn2_x_norm_buf[ALEXNET_STATIC_MAX_BATCH * BN2_UNITS];
float bn3_x_norm_buf[ALEXNET_STATIC_MAX_BATCH * BN3_UNITS];
float bn4_x_norm_buf[ALEXNET_STATIC_MAX_BATCH * BN4_UNITS];
float bn5_x_norm_buf[ALEXNET_STATIC_MAX_BATCH * BN5_UNITS];

// Scratch sized for the largest BN layer (bn1/bn2: C1 channels × 32×32 spatial)
float bn_dxnorm_scratch[ALEXNET_STATIC_MAX_BATCH * BN_MAX_UNITS];

static void bind_bn_layer_buffers(batch_norm_op *op)
{
    switch (op->layer_id) {
        case 1:
            op->avg    = bn1_avg_buf;
            op->var    = bn1_var_buf;
            op->x_norm = bn1_x_norm_buf;
            op->running_mean = bn1_run_mean_buf;
            op->running_var  = bn1_run_var_buf;
            break;
        case 2:
            op->avg    = bn2_avg_buf;
            op->var    = bn2_var_buf;
            op->x_norm = bn2_x_norm_buf;
            op->running_mean = bn2_run_mean_buf;
            op->running_var  = bn2_run_var_buf;
            break;
        case 3:
            op->avg    = bn3_avg_buf;
            op->var    = bn3_var_buf;
            op->x_norm = bn3_x_norm_buf;
            op->running_mean = bn3_run_mean_buf;
            op->running_var  = bn3_run_var_buf;
            break;
        case 4:
            op->avg    = bn4_avg_buf;
            op->var    = bn4_var_buf;
            op->x_norm = bn4_x_norm_buf;
            op->running_mean = bn4_run_mean_buf;
            op->running_var  = bn4_run_var_buf;
            break;
        case 5:
            op->avg    = bn5_avg_buf;
            op->var    = bn5_var_buf;
            op->x_norm = bn5_x_norm_buf;
            op->running_mean = bn5_run_mean_buf;
            op->running_var  = bn5_run_var_buf;
            break;
        default:
            printf_("Error: invalid batchnorm layer_id=%d\n", op->layer_id);
            exit(1);
    }
}


void batch_norm_op_forward(batch_norm_op *op)
{
    register float *input  = op->input;
    register float *output = op->output;

    bind_bn_layer_buffers(op);
    if (op->batchsize > ALEXNET_STATIC_MAX_BATCH) {
        printf_("Error: BN batchsize %d exceeds static max %d\n", op->batchsize, ALEXNET_STATIC_MAX_BATCH);
        exit(1);
    }

    // Inference: normalise with the frozen running stats. No batch statistics
    // are computed and no x_norm is stored (there is no backward pass at eval),
    // so a sample's output is independent of the rest of its batch.
    if (!op->is_training) {
        for (int n = 0; n < op->batchsize; n++) {
            for (int c = 0; c < op->channels; c++) {
                float inv_std = 1.0f / sqrtf(op->running_var[c] + EPSILON);
                float scale = op->gamma[c] * inv_std;
                float shift = op->beta[c] - (op->running_mean[c] * scale);
                
                unsigned long int s = op->spatial_size;

                while (s > 0) {
                    unsigned long int vl;

                    asm volatile("vsetvli %0, %1, e32, m8, ta, ma" : "=r"(vl) : "r"(s));

                    asm volatile("vle32.v v8, (%0)" : : "r"(input));

                    asm volatile("vfmv.v.f v16, %0" : : "f"(shift));

                    asm volatile("vfmacc.vf v16, %0, v8" : : "f"(scale));

                    asm volatile("vse32.v v16, (%0)" : : "r"(output));


                    input += vl;
                    output += vl;
                    s -= vl;
                }
            }
        }
        return;
    }

    memset_vectorized_zero_f32(op->avg, (size_t)op->channels);
    memset_vectorized_zero_f32(op->var, (size_t)op->channels);

    for (int p = 0; p < op->batchsize; p++) {
        for (int c = 0; c < op->channels; c++) {
            unsigned long int s = op->spatial_size;
            float sum = 0.0f;
            size_t vlmax_s;
            asm volatile("vsetvli %0, zero, e32, m8, ta, ma" : "=r"(vlmax_s));
            asm volatile("vfmv.s.f v16, %0" : : "f"(sum));

            while (s > 0) {
                unsigned long int vl;
                asm volatile("vsetvli %0, %1, e32, m8, ta, ma" : "=r"(vl) : "r"(s));
                asm volatile("vle32.v v8, (%0)" : : "r"(input));
                
                // Unordered Float Reduction Sum: v16 = sum(v8) + v16
                asm volatile("vfredusum.vs v16, v8, v16");
                input += vl;
                s -= vl;
            }
            asm volatile("vfmv.f.s %0, v16" : "=f"(sum));
            
            op->avg[c] += sum;
        }
    }
    
    float factor = 1.0f / (op->batchsize * op->spatial_size);
    int c_count = op->channels;
    float *avg_ptr = op->avg;

    while (c_count > 0) {
        unsigned long int vl;
        asm volatile("vsetvli %0, %1, e32, m8, ta, ma" : "=r"(vl) : "r"(c_count));
        
        asm volatile("vle32.v v8, (%0)" : : "r"(avg_ptr));
        asm volatile("vfmul.vf v8, v8, %0" : : "f"(factor));
        asm volatile("vse32.v v8, (%0)" : : "r"(avg_ptr));
        
        avg_ptr += vl;
        c_count -= vl;
    }

    // The mean pass advanced `input` to the end of the tensor; restart it before
    // the variance pass (the scalar version reset its offset here).
    input = op->input;

    for (int p = 0; p < op->batchsize; p++) {
        for (int c = 0; c < op->channels; c++) {
            unsigned long int s = op->spatial_size;
            float var_sum = 0.0f;
            float mean = op->avg[c];
            
            asm volatile("vfmv.s.f v16, %0" : : "f"(var_sum));

            while (s > 0) {
                unsigned long int vl;
                asm volatile("vsetvli %0, %1, e32, m8, ta, ma" : "=r"(vl) : "r"(s));
                asm volatile("vle32.v v8, (%0)" : : "r"(input));
                
                asm volatile("vfsub.vf v8, v8, %0" : : "f"(mean));
                asm volatile("vfmul.vv v8, v8, v8");
                asm volatile("vfredusum.vs v16, v8, v16");
                
                input += vl;
                s -= vl;
            }
            
            asm volatile("vfmv.f.s %0, v16" : "=f"(var_sum));
            
            op->var[c] += var_sum;
        }
    }
    
    float *var_ptr = op->var;

    // Update the running stats from this batch (EMA). PyTorch stores the
    // UNBIASED batch variance in running_var, so apply Bessel's correction
    // N/(N-1); the batch is still normalised below with the biased var above.
    float N = (float)(op->batchsize * op->spatial_size);
    float bessel = (N > 1.0f) ? N / (N - 1.0f) : 1.0f;

    float momentum = BN_MOMENTUM;
    float one_minus_mom = 1.0f - momentum;
    float mom_x_bessel = momentum * bessel;

    // `factor` (= 1/N) is already computed above and reused here.
    // Reset the per-channel cursors: the mean loop advanced avg_ptr to the end.
    c_count = op->channels;
    avg_ptr = op->avg;
    float *rm_ptr = op->running_mean;
    float *rv_ptr = op->running_var;

    while (c_count > 0) {
        unsigned long int vl;
        asm volatile("vsetvli %0, %1, e32, m4, ta, ma" : "=r"(vl) : "r"(c_count));

        asm volatile("vle32.v v4, (%0)" : : "r"(avg_ptr));
        asm volatile("vle32.v v8, (%0)" : : "r"(var_ptr));
        asm volatile("vle32.v v12, (%0)" : : "r"(rm_ptr));
        asm volatile("vle32.v v16, (%0)" : : "r"(rv_ptr));

        asm volatile("vfmul.vf v8, v8, %0" : : "f"(factor));
        asm volatile("vse32.v v8, (%0)" : : "r"(var_ptr));

        asm volatile("vfmul.vf v12, v12, %0" : : "f"(one_minus_mom));
        asm volatile("vfmacc.vf v12, %0, v4" : : "f"(momentum));

        asm volatile("vfmul.vf v16, v16, %0" : : "f"(one_minus_mom));
        asm volatile("vfmacc.vf v16, %0, v8" : : "f"(mom_x_bessel)); 

        asm volatile("vse32.v v12, (%0)" : : "r"(rm_ptr));
        asm volatile("vse32.v v16, (%0)" : : "r"(rv_ptr));

        avg_ptr += vl;
        var_ptr += vl;
        rm_ptr += vl;
        rv_ptr += vl;
        c_count -= vl;
    }

    // Normalise + scale/shift (vectorized below). Restart from the base
    // pointers: the mean/variance passes advanced `input` to the end.
    float *in_ptr    = op->input;
    float *xnorm_ptr = op->x_norm;
    float *out_ptr   = op->output;

    for (int n = 0; n < op->batchsize; n++) {
        for (int cc = 0; cc < op->channels; cc++) {
            
            float mean = op->avg[cc];
            float inv_std = 1.0f / sqrtf(op->var[cc] + EPSILON);
            float g = op->gamma[cc];
            float b = op->beta[cc];
            
            unsigned long int s = op->spatial_size;

            while (s > 0) {
                unsigned long int vl;
                asm volatile("vsetvli %0, %1, e32, m8, ta, ma" : "=r"(vl) : "r"(s));
                
                asm volatile("vle32.v v8, (%0)" : : "r"(in_ptr));
                
                asm volatile("vfsub.vf v8, v8, %0" : : "f"(mean));
                asm volatile("vfmul.vf v8, v8, %0" : : "f"(inv_std));
                
                asm volatile("vse32.v v8, (%0)" : : "r"(xnorm_ptr));
                
                asm volatile("vfmul.vf v16, v8, %0" : : "f"(g));
                asm volatile("vfadd.vf v16, v16, %0" : : "f"(b));
                
                asm volatile("vse32.v v16, (%0)" : : "r"(out_ptr));
                
                in_ptr += vl;
                xnorm_ptr += vl;
                out_ptr += vl;
                s -= vl;
            }
        }
    }
}


void batch_norm_op_backward(batch_norm_op *op)
{
    batch_norm_op_backward_full(op);
}

void batch_norm_op_backward_full(batch_norm_op *op)
{
    int channels     = op->channels;
    int spatial_size = op->spatial_size;

    memset_vectorized_zero_f32(op->d_gamma, (size_t)channels);
    memset_vectorized_zero_f32(op->d_beta,  (size_t)channels);


    // dL/dgamma_c = sum over batch AND spatial of (d_output * x_norm). No 1/B
    // pass follows: cross_entropy_loss already scaled d_output by 1/batchsize,
    // so the sum is the mean gradient. Note this is a plain sum over spatial
    // too — dividing by M (= batchsize * spatial) would shrink gamma/beta by a
    // further factor of spatial_size, which for the conv BNs (spatial 1024 /
    // 256) leaves them effectively frozen.
    float *xnorm_ptr = op->x_norm;
    float *dout_ptr  = op->d_output;

    for (int n = 0; n < op->batchsize; n++) {
        for (int c = 0; c < channels; c++) {
            unsigned long int s = spatial_size;
            
            float dg_sum = 0.0f;
            float db_sum = 0.0f;

            // K6: v25 -> v16. Ara's OPFVV reduction decode (ara_dispatcher.sv)
            // checks vs1 against the ACTIVE lmul (m8 here) via lmul_vs1, which
            // defaults to csr_vtype_q.vlmul and is never reset for VFREDUSUM --
            // unlike vd (protected by a global emul=LMUL_1 default) and unlike
            // the integer reduction path (OPMVV), which doesn't check vs1
            // alignment at all. Per RVV 1.0 a reduction's vs1 is EMUL=1 with no
            // alignment requirement, so this is a genuine Ara bug, not a
            // real constraint -- v25 & 0b111 = 1 != 0 trapped as illegal
            // (confirmed via mepc/mtval read from the trap registers while
            // halted, not guessed). v16 is 8-aligned and free in this loop's
            // scope (confirmed: grepped 313-365 for v16, zero hits before this
            // change) -- the ONLY other user of v16 in this function is the
            // second loop below (line ~395), which loads it fresh via vle32.v
            // strictly after this loop's own v16 has already been read back
            // (below), so there is no overlap. Renamed consistently at every
            // reference (init here, the vfredusum.vs use, and the vfmv.f.s
            // readback) -- renaming only the crashing line would leave the
            // init/readback on the old register and silently corrupt d_beta.
            asm volatile("vfmv.s.f v24, %0" : : "f"(dg_sum)); // Accumulator  d_gamma
            asm volatile("vfmv.s.f v16, %0" : : "f"(db_sum)); // Accumulator  d_beta

            while (s > 0) {
                unsigned long int vl;
                asm volatile("vsetvli %0, %1, e32, m8, ta, ma" : "=r"(vl) : "r"(s));
                
                asm volatile("vle32.v v0, (%0)" : : "r"(xnorm_ptr));
                asm volatile("vle32.v v8, (%0)" : : "r"(dout_ptr));
                
                asm volatile("vfredusum.vs v16, v8, v16");

                asm volatile("vfmul.vv v0, v0, v8");

                asm volatile("vfredusum.vs v24, v0, v24");

                xnorm_ptr += vl;
                dout_ptr += vl;
                s -= vl;
            }

            asm volatile("vfmv.f.s %0, v24" : "=f"(dg_sum));
            asm volatile("vfmv.f.s %0, v16" : "=f"(db_sum));
            
            op->d_gamma[c] += dg_sum;
            op->d_beta[c] += db_sum;
        }
    }

    // Restart the cursors: the d_gamma/d_beta pass advanced them to the end.
    xnorm_ptr = op->x_norm;
    dout_ptr  = op->d_output;
    float *din_ptr = op->d_input;

    // d_gamma/d_beta are now plain sums over batch AND spatial (the 1/batchsize
    // lives in cross_entropy_loss). The dx term needs them divided by
    // M = batchsize*spatial, which is what the scalar reference's  does.
    // Using 1/spatial here would leave dx a factor of batchsize too large --
    // and because this layer's dx feeds the next layer down, that error
    // compounds once per BN in the stack.
    float inv_M = 1.0f / ((float)op->batchsize * (float)spatial_size);

    for (int n = 0; n < op->batchsize; n++) {
        for (int c = 0; c < channels; c++) {
            
            float inv_std = 1.0f / sqrtf(op->var[c] + EPSILON);
            float factor1 = inv_std * op->gamma[c];            // (inv_std * gamma_c)
            float factor2 = op->d_beta[c] * inv_M;            // (d_beta / M)
            float factor3 = op->d_gamma[c] * inv_M;           // (d_gamma / M)
            
            unsigned long int s = spatial_size;

            while (s > 0) {
                unsigned long int vl;
                asm volatile("vsetvli %0, %1, e32, m8, ta, ma" : "=r"(vl) : "r"(s));
                
                asm volatile("vle32.v v8, (%0)"  : : "r"(xnorm_ptr));
                asm volatile("vle32.v v16, (%0)" : : "r"(dout_ptr));
                
                asm volatile("vfmul.vf v24, v8, %0" : : "f"(factor3));
                
                asm volatile("vfadd.vf v24, v24, %0" : : "f"(factor2));
                
                asm volatile("vfsub.vv v16, v16, v24");
                
                asm volatile("vfmul.vf v16, v16, %0" : : "f"(factor1));
                
                asm volatile("vse32.v v16, (%0)" : : "r"(din_ptr));
                
                xnorm_ptr += vl;
                dout_ptr  += vl;
                din_ptr   += vl;
                s -= vl;
            }
        }
    }
}
void batch_norm_op_backward_input_only(batch_norm_op *op)
{
    int channels     = op->channels;
    int spatial_size = op->spatial_size;
    float M = (float)(op->batchsize * spatial_size);

    float S1[BN_MAX_CHANNELS];
    float S2[BN_MAX_CHANNELS];
    memset_vectorized_zero_f32(S1, (size_t)channels);
    memset_vectorized_zero_f32(S2, (size_t)channels);

    float *xnorm_ptr = op->x_norm;
    float *dout_ptr  = op->d_output;

    for (int n = 0; n < op->batchsize; n++) {
        for (int c = 0; c < channels; c++) {
            unsigned long int s = spatial_size;
            float sum_dy = 0.0f;
            float sum_dy_xn = 0.0f;

            // K6: v25 -> v16, same reasoning and same safety check as the
            // identical pattern in batch_norm_op_backward_full above (grepped
            // this function's 415-462 scope for v16: zero hits before this
            // change; the only other v16 user is the second loop below,
            // sequentially after this one's own readback).
            asm volatile("vfmv.s.f v24, %0" : : "f"(sum_dy_xn));
            asm volatile("vfmv.s.f v16, %0" : : "f"(sum_dy));

            while (s > 0) {
                unsigned long int vl;
                asm volatile("vsetvli %0, %1, e32, m8, ta, ma" : "=r"(vl) : "r"(s));
                
                asm volatile("vle32.v v0, (%0)" : : "r"(xnorm_ptr));
                asm volatile("vle32.v v8, (%0)" : : "r"(dout_ptr));
                
                asm volatile("vfredusum.vs v16, v8, v16");

                asm volatile("vfmul.vv v0, v0, v8");

                asm volatile("vfredusum.vs v24, v0, v24");

                xnorm_ptr += vl;
                dout_ptr += vl;
                s -= vl;
            }

            asm volatile("vfmv.f.s %0, v24" : "=f"(sum_dy_xn));
            asm volatile("vfmv.f.s %0, v16" : "=f"(sum_dy));
            
            S1[c] += sum_dy;
            S2[c] += sum_dy_xn;
        }
    }

    xnorm_ptr = op->x_norm;
    dout_ptr  = op->d_output;
    float *din_ptr   = op->d_input;
    float inv_M = 1.0f / M;

    for (int n = 0; n < op->batchsize; n++) {
        for (int c = 0; c < channels; c++) {
            
            float inv_std = 1.0f / sqrtf(op->var[c] + EPSILON);
            float factor1 = inv_std * op->gamma[c]; 
            float factor2 = S1[c] * inv_M; //   d_beta / spatial_size
            float factor3 = S2[c] * inv_M; //   d_gamma / spatial_size

            unsigned long int s = spatial_size;

            while (s > 0) {
                unsigned long int vl;
                asm volatile("vsetvli %0, %1, e32, m8, ta, ma" : "=r"(vl) : "r"(s));
                
                asm volatile("vle32.v v8, (%0)"  : : "r"(xnorm_ptr));
                asm volatile("vle32.v v16, (%0)" : : "r"(dout_ptr));
                
                asm volatile("vfmul.vf v24, v8, %0" : : "f"(factor3));
                
                asm volatile("vfadd.vf v24, v24, %0" : : "f"(factor2));
                
                asm volatile("vfsub.vv v16, v16, v24");
                
                asm volatile("vfmul.vf v16, v16, %0" : : "f"(factor1));
                
                asm volatile("vse32.v v16, (%0)" : : "r"(din_ptr));
                
                xnorm_ptr += vl;
                dout_ptr  += vl;
                din_ptr   += vl;
                s -= vl;
            }
        }
    }
}


void calloc_batchnorm_weights(batch_norm_op *op)
{
    if (op->gamma)
        memset_vectorized_zero_f32(op->gamma, (size_t)op->channels);
    if (op->beta)
        memset_vectorized_zero_f32(op->beta, (size_t)op->channels);
}

void free_batchnorm_weights(batch_norm_op *op)
{
    (void)op;
}

void calloc_batchnorm_dweights(batch_norm_op *op)
{
    if (op->d_gamma)
        memset_vectorized_zero_f32(op->d_gamma, (size_t)op->channels);
    if (op->d_beta)
        memset_vectorized_zero_f32(op->d_beta, (size_t)op->channels);
}

void free_batchnorm_dweights(batch_norm_op *op)
{
    (void)op;
}

void save_batchnorm_weights(batch_norm_op *op)
{
    (void)op;
}

void load_batchnorm_weights(batch_norm_op *op, const float *gamma_array, const float *beta_array)
{
    memcpy_vectorized_f32(op->gamma, gamma_array, (size_t)op->channels);
    memcpy_vectorized_f32(op->beta,  beta_array,  (size_t)op->channels);
}

void batchnorm_reset_running_stats(void)
{
    struct { float *mean; float *var; int n; } bn[] = {
        { bn1_run_mean_buf, bn1_run_var_buf, BN1_CHANNELS },
        { bn2_run_mean_buf, bn2_run_var_buf, BN2_CHANNELS },
        { bn3_run_mean_buf, bn3_run_var_buf, BN3_CHANNELS },
        { bn4_run_mean_buf, bn4_run_var_buf, BN4_CHANNELS },
        { bn5_run_mean_buf, bn5_run_var_buf, BN5_CHANNELS },
    };

    float zero = 0.0f;
    float one  = 1.0f;

    for (int k = 0; k < 5; k++) {
        int c_count  = bn[k].n;
        float *m_ptr = bn[k].mean;
        float *v_ptr = bn[k].var;

        while (c_count > 0) {
            unsigned long int vl;
            asm volatile("vsetvli %0, %1, e32, m8, ta, ma" : "=r"(vl) : "r"(c_count));

            asm volatile("vfmv.v.f v8, %0" : : "f"(zero));
            
            asm volatile("vfmv.v.f v16, %0" : : "f"(one));

            asm volatile("vse32.v v8, (%0)" : : "r"(m_ptr));
            asm volatile("vse32.v v16, (%0)" : : "r"(v_ptr));

            m_ptr += vl;
            v_ptr += vl;
            c_count -= vl;
        }
    }
}