//
// File:        matrix.c
// Description: Implementation of matrix computation
// Author:      Haris Wang
// Modified: Stavros Mitropoulos
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "alexnet.h"
#include "fmatmul.h"
// #include <immintrin.h> 
#ifdef SPIKE
#include <printf.h>
#elif defined ARA_LINUX
#include <stdio.h>
#else
#include "printf.h"
#endif

#define MATRIX_TRANSPOSE_WORKSPACE_ELEMS (FC_MAX_IN_UNITS * FC_MAX_INTERNAL)

float shared_memory_pool[MATRIX_TRANSPOSE_WORKSPACE_ELEMS];

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

// float fmatmul_a_scratch[FMATMUL_MAX_M * FMATMUL_MAX_N];
float fmatmul_c_scratch[FMATMUL_MAX_M * FMATMUL_MAX_K];
// FP16 output scratch for the hidden-layer forward path (fmatmul_fused_f16out).
_Float16 fmatmul_c_scratch_f16[FMATMUL_MAX_M * FMATMUL_MAX_K];
static float fmatmul_nt_loop_scratch[FMATMUL_MAX_M * FMATMUL_MAX_K];
static float fmatmul_nt_out_scratch[FMATMUL_MAX_M * FMATMUL_MAX_K];

static void matrix_multiply_scalar(const _Float16 *a, const _Float16 *b, float *c,
                                   const int M, const int N, const int K);
static void matrix_multiply_scalar_nt(const _Float16 *a, const _Float16 *b, float *c,
                                      const int M, const int N, const int K);
static void matrix_multiply_scalar_tn(const _Float16 *a, const _Float16 *b, float *c,
                                      const int M, const int N, const int K);
static void matrix_multiply_scalar_fused(const _Float16 *a, const _Float16 *b,
                                         const float *bias, float *c,
                                         const int M, const int N,
                                         const int K);

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


void matrix_multiply(const _Float16 *a, const _Float16 *b, float *c, const int M, const int N, const int K)
{
    /**
     * matrix multiply, c = a * b
     *
     * Input:
     * a    [M,N]
     * b    [N,K]
     * Output:
     * c    [M,K]
     * */
    if (M <= 0 || N <= 0 || K <= 0)
        return;
    _Float16 *fmatmul_a_scratch = (_Float16 *)shared_memory_pool;
    unsigned long int block = fmatmul_row_block((unsigned long int)M);
    unsigned long int padded_m = (((unsigned long int)M + block - 1) / block) * block;

    if ((unsigned long int)N > FMATMUL_MAX_N ||
        (unsigned long int)K > FMATMUL_MAX_K ||
        padded_m > FMATMUL_MAX_M)
    {
        printf_("[SCALAR] matrix_multiply: M=%d N=%d K=%d padded_m=%lu MAX_M=%d\n", M, N, K, padded_m, FMATMUL_MAX_M);
        matrix_multiply_scalar(a, b, c, M, N, K);
        return;
    }

    if (padded_m == (unsigned long int)M) {
        fmatmul(c, a, b,
                (unsigned long int)M, (unsigned long int)N, (unsigned long int)K);
        return;
    }

    const size_t mn = (size_t)M * (size_t)N;
    const size_t pnk = (size_t)padded_m * (size_t)N;
    const size_t mk = (size_t)M * (size_t)K;

    size_t remaining_mn = mn;
    const _Float16 *src_mn = a;
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

    fmatmul(fmatmul_c_scratch, fmatmul_a_scratch, b,
            padded_m, (unsigned long int)N, (unsigned long int)K);

    size_t remaining_mk = mk;
    const float *src_mk = fmatmul_c_scratch;
    float *dst_mk = c;
    while (remaining_mk > 0)
    {
        size_t vl = 0;
        asm volatile("vsetvli %0, %1, e32, m1, ta, ma" : "=r"(vl) : "r"(remaining_mk));
        asm volatile("vle32.v v0, (%0);" : : "r"(src_mk) : "memory");
        asm volatile("vle32.v v8, (%0);" : : "r"(dst_mk) : "memory");
        asm volatile("vfadd.vv v8, v8, v0");
        asm volatile("vse32.v v8, (%0);" : : "r"(dst_mk) : "memory");
        src_mk += vl;
        dst_mk += vl;
        remaining_mk -= vl;
    }
}

void matrix_multiply_fused(const _Float16 *a, const _Float16 *b, const float *bias,
                           float *c, const int M, const int N, const int K)
{
    /**
     * matrix multiply with fused bias, c = a * b + bias
     *
     * Input:
     * a    [M,N]
     * b    [N,K]
     * bias [K]
     * Output:
     * c    [M,K]
     * */
    if (M <= 0 || N <= 0 || K <= 0)
        return;

    _Float16 *fmatmul_a_scratch = (_Float16 *)shared_memory_pool;
    unsigned long int block = fmatmul_row_block((unsigned long int)M);
    unsigned long int padded_m = (((unsigned long int)M + block - 1) / block) * block;

    if ((unsigned long int)N > FMATMUL_MAX_N ||
        (unsigned long int)K > FMATMUL_MAX_K ||
        padded_m > FMATMUL_MAX_M)
    {
        printf_("[SCALAR] matrix_multiply_fused: M=%d N=%d K=%d padded_m=%lu MAX_M=%d\n", M, N, K, padded_m, FMATMUL_MAX_M);
        matrix_multiply_scalar_fused(a, b, bias, c, M, N, K);
        return;
    }

    if (padded_m == (unsigned long int)M) {
        fmatmul_fused(c, a, b, bias,
                      (unsigned long int)M, (unsigned long int)N, (unsigned long int)K);
        return;
    }

    const size_t mn = (size_t)M * (size_t)N;
    const size_t pnk = (size_t)padded_m * (size_t)N;
    const size_t mk = (size_t)M * (size_t)K;

    size_t remaining_mn = mn;
    const _Float16 *src_mn = a;
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

    fmatmul_fused(fmatmul_c_scratch, fmatmul_a_scratch, b, bias,
                 padded_m, (unsigned long int)N, (unsigned long int)K);

    size_t remaining_mk = mk;
    const float *src_mk = fmatmul_c_scratch;
    float *dst_mk = c;
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
}

void matrix_multiply_nt(const _Float16 *a, const _Float16 *b, float *c,
                        const int M, const int N, const int K)
{
    /**
     * matrix multiply, c += a * b^T
     *
     * Input:
     * a    [M,N]
     * b    [K,N]
     * Output:
     * c    [M,K]
     * */
    if (M <= 0 || N <= 0 || K <= 0)
        return;

    _Float16 *fmatmul_a_scratch = (_Float16 *)shared_memory_pool;
    unsigned long int block = fmatmul_row_block((unsigned long int)M);
    unsigned long int padded_m = (((unsigned long int)M + block - 1) / block) * block;

    // if ((unsigned long int)N > FMATMUL_MAX_N ||
    //     (unsigned long int)K > FMATMUL_MAX_K ||
    //     padded_m > FMATMUL_MAX_M)
    // {
    //     matrix_multiply_scalar_nt(a, b, c, M, N, K);
    //     return;
    // }

    const size_t mn = (size_t)M * (size_t)N;
    const size_t pnk = (size_t)padded_m * (size_t)N;
    const size_t mk = (size_t)M * (size_t)K;

    if (padded_m == (unsigned long int)M) {
        fmatmul_nt(fmatmul_c_scratch, a, b,
                   (unsigned long int)M, (unsigned long int)N, (unsigned long int)K);
    } else {
        size_t remaining_mn = mn;
        const _Float16 *src_mn = a;
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

        fmatmul_nt(fmatmul_c_scratch, fmatmul_a_scratch, b,
                   padded_m, (unsigned long int)N, (unsigned long int)K);
    }

    size_t remaining_mk = mk;
    const float *src_mk = fmatmul_c_scratch;
    float *dst_mk = c;
    while (remaining_mk > 0)
    {
        size_t vl = 0;
        asm volatile("vsetvli %0, %1, e32, m1, ta, ma" : "=r"(vl) : "r"(remaining_mk));
        asm volatile("vle32.v v0, (%0);" : : "r"(src_mk) : "memory");
        asm volatile("vle32.v v8, (%0);" : : "r"(dst_mk) : "memory");
        asm volatile("vfadd.vv v8, v8, v0");
        asm volatile("vse32.v v8, (%0);" : : "r"(dst_mk) : "memory");
        src_mk += vl;
        dst_mk += vl;
        remaining_mk -= vl;
    }
}

// FP16-output variant of matrix_multiply_nt: c = a * b^T with FP16 c (d_input).
// Overwrite semantics (not accumulate): the caller pre-zeroes d_input and calls
// this once, so overwrite == accumulate-from-zero. d_input carries the loss scale
// into the previous layer's FP16 gradient chain (it is NOT unscaled here).
void matrix_multiply_nt_f16out(const _Float16 *a, const _Float16 *b, _Float16 *c,
                               const int M, const int N, const int K)
{
    if (M <= 0 || N <= 0 || K <= 0)
        return;

    _Float16 *fmatmul_a_scratch = (_Float16 *)shared_memory_pool;
    unsigned long int block = fmatmul_row_block((unsigned long int)M);
    unsigned long int padded_m = (((unsigned long int)M + block - 1) / block) * block;

    if (padded_m == (unsigned long int)M) {
        // No row padding: narrow straight into c.
        fmatmul_nt_f16out(c, a, b,
                          (unsigned long int)M, (unsigned long int)N, (unsigned long int)K);
        return;
    }

    const size_t mn = (size_t)M * (size_t)N;
    const size_t pnk = (size_t)padded_m * (size_t)N;
    const size_t mk = (size_t)M * (size_t)K;

    // Pack a into the padded FP16 scratch (rows [M, padded_m) zero-filled).
    size_t remaining_mn = mn;
    const _Float16 *src_mn = a;
    _Float16 *dst_mn = fmatmul_a_scratch;
    while (remaining_mn > 0) {
        size_t vl = 0;
        asm volatile("vsetvli %0, %1, e16, m1, ta, ma" : "=r"(vl) : "r"(remaining_mn));
        asm volatile("vle16.v v0, (%0);" : : "r"(src_mn) : "memory");
        asm volatile("vse16.v v0, (%0);" : : "r"(dst_mn) : "memory");
        src_mn += vl; dst_mn += vl; remaining_mn -= vl;
    }
    size_t remaining = pnk - mn;
    _Float16 *dstz = fmatmul_a_scratch + mn;
    while (remaining > 0) {
        size_t vl = 0;
        asm volatile("vsetvli %0, %1, e16, m1, ta, ma" : "=r"(vl) : "r"(remaining));
        asm volatile("vmv.v.x v0, zero");
        asm volatile("vse16.v v0, (%0);" : : "r"(dstz) : "memory");
        dstz += vl; remaining -= vl;
    }

    fmatmul_nt_f16out(fmatmul_c_scratch_f16, fmatmul_a_scratch, b,
                      padded_m, (unsigned long int)N, (unsigned long int)K);

    // Copy the valid M*K FP16 rows into c.
    size_t remaining_mk = mk;
    const _Float16 *src_mk = fmatmul_c_scratch_f16;
    _Float16 *dst_mk = c;
    while (remaining_mk > 0) {
        size_t vl = 0;
        asm volatile("vsetvli %0, %1, e16, m1, ta, ma" : "=r"(vl) : "r"(remaining_mk));
        asm volatile("vle16.v v0, (%0);" : : "r"(src_mk) : "memory");
        asm volatile("vse16.v v0, (%0);" : : "r"(dst_mk) : "memory");
        src_mk += vl; dst_mk += vl; remaining_mk -= vl;
    }
}

void matrix_multiply_tn(const _Float16 *a, const _Float16 *b, float *c,
                        const int M, const int N, const int K)
{
    /**
     * matrix multiply, c += a^T * b
     *
     * Input:
     * a    [M,N]
     * b    [M,K]
     * Output:
     * c    [N,K]
     * */
    if (M <= 0 || N <= 0 || K <= 0)
        return;

    unsigned long int block = fmatmul_row_block((unsigned long int)N);
    unsigned long int padded_n = (((unsigned long int)N + block - 1) / block) * block;

    if (padded_n == (unsigned long int)N) {
        fmatmul_tn(c, a, b,
                   (unsigned long int)M, (unsigned long int)N, (unsigned long int)K);
        return;
    }

    if (((size_t)M * (size_t)padded_n) > MATRIX_TRANSPOSE_WORKSPACE_ELEMS) {
        printf_("[SCALAR] matrix_multiply_tn: M=%d N=%d K=%d (workspace too small)\n", M, N, K);
        matrix_multiply_scalar_tn(a, b, c, M, N, K);
        return;
    }

    _Float16 *a_pad = (_Float16 *)shared_memory_pool;
    for (int m = 0; m < M; m++)
    {
        const _Float16 *src_row = a + (size_t)m * (size_t)N;
        _Float16 *dst_row = a_pad + (size_t)m * (size_t)padded_n;

        size_t remaining_row = (size_t)N;
        while (remaining_row > 0)
        {
            size_t vl = 0;
            asm volatile("vsetvli %0, %1, e16, m1, ta, ma" : "=r"(vl) : "r"(remaining_row));
            asm volatile("vle16.v v0, (%0);" : : "r"(src_row) : "memory");
            asm volatile("vse16.v v0, (%0);" : : "r"(dst_row) : "memory");
            src_row += vl;
            dst_row += vl;
            remaining_row -= vl;
        }

        size_t remaining_pad = (size_t)padded_n - (size_t)N;
        while (remaining_pad > 0)
        {
            size_t vl = 0;
            asm volatile("vsetvli %0, %1, e16, m1, ta, ma" : "=r"(vl) : "r"(remaining_pad));
            asm volatile("vmv.v.x v0, zero");
            asm volatile("vse16.v v0, (%0);" : : "r"(dst_row) : "memory");
            dst_row += vl;
            remaining_pad -= vl;
        }
    }

    fmatmul_tn(c, a_pad, b,
               (unsigned long int)M, (unsigned long int)N, (unsigned long int)K);
}

int matrix_multiply_tn_verify(const _Float16 *a, const _Float16 *b,
                              const int M, const int N, const int K,
                              const float eps)
{
    if (M <= 0 || N <= 0 || K <= 0)
        return 0;

    const size_t nk = (size_t)N * (size_t)K;
    if (nk > MATRIX_TRANSPOSE_WORKSPACE_ELEMS) {
        printf_("matrix_multiply_tn_verify: dims too large (%d x %d)\n", N, K);
        return 0;
    }

    float *tn_out = shared_memory_pool;
    for (size_t idx = 0; idx < nk; idx++)
        tn_out[idx] = 0.0f;

    fmatmul_tn(tn_out, a, b,
               (unsigned long int)M, (unsigned long int)N, (unsigned long int)K);

    size_t mismatch_count = 0;
    float max_diff = 0.0f;
    const float inv_batch = 1.0f / (float)M;

    for (int n = 0; n < N; n++)
    {
        for (int k = 0; k < K; k++)
        {
            float sum = 0.0f;
            for (int m = 0; m < M; m++)
                sum += a[(size_t)m * (size_t)N + (size_t)n] *
                       b[(size_t)m * (size_t)K + (size_t)k];
            sum *= inv_batch;

            float diff = fabsf(sum - tn_out[(size_t)n * (size_t)K + (size_t)k]);
            if (diff > max_diff)
                max_diff = diff;
            if (diff > eps)
                mismatch_count++;
        }
    }

    if (mismatch_count == 0)
        printf_("matrix_multiply_tn_verify: OK (max diff %f)\n", max_diff);
    else
        printf_("matrix_multiply_tn_verify: FAIL (mismatches %lu, max diff %f)\n",
                (unsigned long)mismatch_count, max_diff);

    return (mismatch_count == 0) ? 1 : 0;
}
void matrix_multiply_nt_deferred(const _Float16 *a, const _Float16 *b, float *c,
                                 const int M, const int N, const int K)
{
    if (M <= 0 || N <= 0 || K <= 0) return;

    if (((unsigned long int)K % 4) != 0) {
        matrix_multiply_nt(a, b, c, M, N, K);
        return;
    }

    _Float16 *fmatmul_a_scratch = (_Float16 *)shared_memory_pool;
    unsigned long int padded_m = (((unsigned long int)M + 3) / 4) * 4;

    const size_t mn = (size_t)M * (size_t)N;
    const size_t pnk = (size_t)padded_m * (size_t)N;
    const size_t mk_pad = (size_t)padded_m * (size_t)K;

    // 1. Vectorized Copy A -> scratch (m8)
    size_t rem = mn;
    const _Float16 *src = a;
    _Float16 *dst = fmatmul_a_scratch;
    while (rem > 0) {
        size_t vl;
        asm volatile("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(rem));
        asm volatile("vle16.v v8, (%0)" :: "r"(src));
        asm volatile("vse16.v v8, (%0)" :: "r"(dst));
        src += vl; dst += vl;
        rem -= vl;
    }

    // 2. Vectorized Zero Padding (a_scratch)
    rem = pnk - mn;
    dst = fmatmul_a_scratch + mn;
    while (rem > 0) {
        size_t vl;
        asm volatile("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(rem));
        asm volatile("vmv.v.i v8, 0");
        asm volatile("vse16.v v8, (%0)" :: "r"(dst));
        dst += vl;
        rem -= vl;
    }

    // 3. Vectorized Zeroing c_scratch (FP32)
    rem = mk_pad;
    float *ptr = fmatmul_c_scratch;
    while (rem > 0) {
        size_t vl;
        asm volatile("vsetvli %0, %1, e32, m8, ta, ma" : "=r"(vl) : "r"(rem));
        asm volatile("vmv.v.i v8, 0");
        asm volatile("vse32.v v8, (%0)" :: "r"(ptr));
        ptr += vl;
        rem -= vl;
    }

    // Call the core kernel
    fmatmul_4x4_deferred(fmatmul_c_scratch, fmatmul_a_scratch, b,
                         padded_m, (unsigned long int)N, (unsigned long int)K);

    // Final reduction
    rem = (size_t)M * (size_t)K;
    float *ptr_c = c;
    const float *ptr_scratch = fmatmul_c_scratch;
    
    while (rem > 0) {
        size_t vl;
        asm volatile("vsetvli %0, %1, e32, m8, ta, ma" : "=r"(vl) : "r"(rem));
        
        // Φόρτωση από το αποτέλεσμα και τον πίνακα c
        asm volatile("vle32.v v8, (%0)" :: "r"(ptr_scratch));
        asm volatile("vle32.v v16, (%0)" :: "r"(ptr_c));
        
        // c[idx] += fmatmul_c_scratch[idx]
        asm volatile("vfadd.vv v16, v16, v8");
        
        // Αποθήκευση πίσω στο c
        asm volatile("vse32.v v16, (%0)" :: "r"(ptr_c));
        
        ptr_c += vl;
        ptr_scratch += vl;
        rem -= vl;
    }
}

int matrix_multiply_nt_verify(const _Float16 *a, const _Float16 *b,
                              const int M, const int N, const int K,
                              const float eps)
{
    if (M <= 0 || N <= 0 || K <= 0)
        return 0;

    // if ((unsigned long int)M > FMATMUL_MAX_M ||
    //     (unsigned long int)K > FMATMUL_MAX_K)
    // {
    //     printf_("matrix_multiply_nt_verify: dims too large (%d x %d)\n", M, K);
    //     return 0;
    // }

    const size_t mk = (size_t)M * (size_t)K;
    for (size_t idx = 0; idx < mk; idx++) {
        fmatmul_nt_loop_scratch[idx] = 0.0f;
        fmatmul_nt_out_scratch[idx] = 0.0f;
    }

    matrix_multiply_scalar_nt(a, b, fmatmul_nt_loop_scratch, M, N, K);
    matrix_multiply_nt(a, b, fmatmul_nt_out_scratch, M, N, K);

    size_t mismatch_count = 0;
    float max_diff = 0.0f;
    for (size_t idx = 0; idx < mk; idx++) {
        float diff = fabsf(fmatmul_nt_loop_scratch[idx] - fmatmul_nt_out_scratch[idx]);
        if (diff > max_diff)
            max_diff = diff;
        if (diff > eps)
            mismatch_count++;
    }

    if (mismatch_count == 0)
        printf_("matrix_multiply_nt_verify: OK (max diff %f)\n", max_diff);
    else
        printf_("matrix_multiply_nt_verify: FAIL (mismatches %lu, max diff %f)\n",
                (unsigned long)mismatch_count, max_diff);

    return (mismatch_count == 0) ? 1 : 0;
}

void matrix_transpose(float *x, int m, int n)
{
    /** matrix transpose
     * 
     * Input:
     *      x[m,n]
     * Output:
     *      x[n,m]
     * */
    size_t elems = (size_t)m * (size_t)n;
    if (elems > MATRIX_TRANSPOSE_WORKSPACE_ELEMS) {
        printf_("Error: matrix_transpose workspace too small for %d x %d\n", m, n);
        exit(1);
    }
    float *tmp = shared_memory_pool;
    register int i, j;
    register float *ptr = x;
    for (i = 0; i < m; i++)
    {
        for (j = 0; j < n; j++)
            tmp[j*m+i] = *(ptr++);
    }
    memcpy(x, tmp, elems * sizeof(float));
    return;
}



static void matrix_multiply_scalar(const _Float16 *a, const _Float16 *b, float *c, const int M, const int N, const int K)
{
    register int i, j, p;
    register const _Float16 *a_ptr = a;
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

static void matrix_multiply_scalar_nt(const _Float16 *a, const _Float16 *b, float *c,
                                      const int M, const int N, const int K)
{
    for (int i = 0; i < M; i++)
    {
        const _Float16 *a_ptr = a + (size_t)i * (size_t)N;
        for (int k = 0; k < K; k++)
        {
            const _Float16 *b_ptr = b + (size_t)k * (size_t)N;
            float sum = 0.0f;
            for (int j = 0; j < N; j++)
                sum += (float)a_ptr[j] * (float)b_ptr[j];
            c[(size_t)i * (size_t)K + (size_t)k] += sum;
        }
    }
}

static void matrix_multiply_scalar_tn(const _Float16 *a, const _Float16 *b, float *c,
                                      const int M, const int N, const int K)
{
    for (int n = 0; n < N; n++)
    {
        for (int k = 0; k < K; k++)
        {
            float sum = 0.0f;
            for (int m = 0; m < M; m++)
                sum += (float)a[(size_t)m * (size_t)N + (size_t)n] *
                       (float)b[(size_t)m * (size_t)K + (size_t)k];
            c[(size_t)n * (size_t)K + (size_t)k] += sum;
        }
    }
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