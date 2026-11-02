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
#include "utils.h"
// #include <immintrin.h> 
#ifdef SPIKE
#include <printf.h>
#elif defined ARA_LINUX
#include <stdio.h>
#else
#include "printf.h"
#endif

#define MATRIX_TRANSPOSE_WORKSPACE_ELEMS (FC_MAX_IN_UNITS * FC_MAX_INTERNAL)

float shared_memory_pool_32[MATRIX_TRANSPOSE_WORKSPACE_ELEMS];

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
float fmatmul_c_scratch_32[FMATMUL_MAX_M * FMATMUL_MAX_K];
static float fmatmul_nt_loop_scratch[FMATMUL_MAX_M * FMATMUL_MAX_K];
static float fmatmul_nt_out_scratch[FMATMUL_MAX_M * FMATMUL_MAX_K];

static void matrix_multiply_scalar(const float *a, const float *b, float *c,
                                   const int M, const int N, const int K);
static void matrix_multiply_scalar_nt(const float *a, const float *b, float *c,
                                      const int M, const int N, const int K);
static void matrix_multiply_scalar_tn(const float *a, const float *b, float *c,
                                      const int M, const int N, const int K);
static void matrix_multiply_scalar_fused(const float *a, const float *b,
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


void matrix_multiply_32(const float *a, const float *b, float *c, const int M, const int N, const int K)
{
    /**
     * matrix multiply, c = a * b   (c is OVERWRITTEN, not accumulated)
     *
     * Input:
     * a    [M,N]
     * b    [N,K]
     * Output:
     * c    [M,K]
     *
     * All three exit paths below must honour that contract. They previously did
     * not: fmatmul_32() overwrites, but the scalar fallback and the M-padding
     * copy-back both ACCUMULATED into c. Which one you got depended on the
     * matrix shape (via fmatmul_row_block() and the FMATMUL_MAX_* limits), so
     * the same call could overwrite or accumulate depending on M/N/K -- and a
     * caller that did not pre-zero c silently summed every invocation on top of
     * the previous one. conv's d_weights loop hits the padding branch (M = oc
     * is 12 -> 16 or 24 -> 32), which is how a missing memset there piled all
     * 32 batch images on top of each other.
     * */
    if (M <= 0 || N <= 0 || K <= 0)
        return;
    float *fmatmul_a_scratch = shared_memory_pool_32;
    unsigned long int block = fmatmul_row_block((unsigned long int)M);
    unsigned long int padded_m = (((unsigned long int)M + block - 1) / block) * block;

    if ((unsigned long int)N > FMATMUL_MAX_N ||
        (unsigned long int)K > FMATMUL_MAX_K ||
        padded_m > FMATMUL_MAX_M)
    {
        printf_("[SCALAR] matrix_multiply_32: M=%d N=%d K=%d padded_m=%lu MAX_M=%d\n", M, N, K, padded_m, FMATMUL_MAX_M);
        // matrix_multiply_scalar() accumulates (`*c_ptr += ...`), so zero c
        // first to keep this path's contract identical to the vector paths.
        memset_vectorized_zero_f32(c, (size_t)M * (size_t)K);
        matrix_multiply_scalar(a, b, c, M, N, K);
        return;
    }

    if (padded_m == (unsigned long int)M) {
        fmatmul_32(c, a, b,
                (unsigned long int)M, (unsigned long int)N, (unsigned long int)K);
        return;
    }

    const size_t mn = (size_t)M * (size_t)N;
    const size_t pnk = (size_t)padded_m * (size_t)N;
    const size_t mk = (size_t)M * (size_t)K;

    size_t remaining_mn = mn;
    const float *src_mn = a;
    float *dst_mn = fmatmul_a_scratch;
    while (remaining_mn > 0)
    {
        size_t vl = 0;
        asm volatile("vsetvli %0, %1, e32, m1, ta, ma" : "=r"(vl) : "r"(remaining_mn));
        asm volatile("vle32.v v0, (%0);" : : "r"(src_mn) : "memory");
        asm volatile("vse32.v v0, (%0);" : : "r"(dst_mn) : "memory");
        src_mn += vl;
        dst_mn += vl;
        remaining_mn -= vl;
    }

    size_t remaining = pnk - mn;
    float *dst = fmatmul_a_scratch + mn;
    while (remaining > 0)
    {
        size_t vl = 0;
        asm volatile("vsetvli %0, %1, e32, m1, ta, ma" : "=r"(vl) : "r"(remaining));
        asm volatile("vmv.v.x v0, zero");
        asm volatile("vse32.v v0, (%0);" : : "r"(dst) : "memory");
        dst += vl;
        remaining -= vl;
    }

    fmatmul_32(fmatmul_c_scratch_32, fmatmul_a_scratch, b,
            padded_m, (unsigned long int)N, (unsigned long int)K);

    size_t remaining_mk = mk;
    const float *src_mk = fmatmul_c_scratch_32;
    float *dst_mk = c;
    while (remaining_mk > 0)
    {
        size_t vl = 0;
        asm volatile("vsetvli %0, %1, e32, m1, ta, ma" : "=r"(vl) : "r"(remaining_mk));
        // Plain copy-back: the M-padded result in fmatmul_c_scratch_32 IS the
        // answer. This used to vle32 the old c and vfadd into it, accumulating.
        asm volatile("vle32.v v0, (%0);" : : "r"(src_mk) : "memory");
        asm volatile("vse32.v v0, (%0);" : : "r"(dst_mk) : "memory");
        src_mk += vl;
        dst_mk += vl;
        remaining_mk -= vl;
    }
}

void matrix_multiply_fused_32(const float *a, const float *b, const float *bias,
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

    float *fmatmul_a_scratch = shared_memory_pool_32;
    unsigned long int block = fmatmul_row_block((unsigned long int)M);
    unsigned long int padded_m = (((unsigned long int)M + block - 1) / block) * block;

    if ((unsigned long int)N > FMATMUL_MAX_N ||
        (unsigned long int)K > FMATMUL_MAX_K ||
        padded_m > FMATMUL_MAX_M)
    {
        printf_("[SCALAR] matrix_multiply_fused_32: M=%d N=%d K=%d padded_m=%lu MAX_M=%d\n", M, N, K, padded_m, FMATMUL_MAX_M);
        matrix_multiply_scalar_fused(a, b, bias, c, M, N, K);
        return;
    }

    if (padded_m == (unsigned long int)M) {
        fmatmul_fused_32(c, a, b, bias,
                      (unsigned long int)M, (unsigned long int)N, (unsigned long int)K);
        return;
    }

    const size_t mn = (size_t)M * (size_t)N;
    const size_t pnk = (size_t)padded_m * (size_t)N;
    const size_t mk = (size_t)M * (size_t)K;

    size_t remaining_mn = mn;
    const float *src_mn = a;
    float *dst_mn = fmatmul_a_scratch;
    while (remaining_mn > 0)
    {
        size_t vl = 0;
        asm volatile("vsetvli %0, %1, e32, m1, ta, ma" : "=r"(vl) : "r"(remaining_mn));
        asm volatile("vle32.v v0, (%0);" : : "r"(src_mn) : "memory");
        asm volatile("vse32.v v0, (%0);" : : "r"(dst_mn) : "memory");
        src_mn += vl;
        dst_mn += vl;
        remaining_mn -= vl;
    }

    size_t remaining = pnk - mn;
    float *dst = fmatmul_a_scratch + mn;
    while (remaining > 0)
    {
        size_t vl = 0;
        asm volatile("vsetvli %0, %1, e32, m1, ta, ma" : "=r"(vl) : "r"(remaining));
        asm volatile("vmv.v.x v0, zero");
        asm volatile("vse32.v v0, (%0);" : : "r"(dst) : "memory");
        dst += vl;
        remaining -= vl;
    }

    fmatmul_fused_32(fmatmul_c_scratch_32, fmatmul_a_scratch, b, bias,
                 padded_m, (unsigned long int)N, (unsigned long int)K);

    size_t remaining_mk = mk;
    const float *src_mk = fmatmul_c_scratch_32;
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

void matrix_multiply_nt_32(const float *a, const float *b, float *c,
                        const int M, const int N, const int K)
{
    /**
     * matrix multiply, c = a * b^T
     *
     * Input:
     * a    [M,N]
     * b    [K,N]
     * Output:
     * c    [M,K]
     * */
    if (M <= 0 || N <= 0 || K <= 0)
        return;

    float *fmatmul_a_scratch = shared_memory_pool_32;
    unsigned long int block = fmatmul_row_block((unsigned long int)M);
    unsigned long int padded_m = (((unsigned long int)M + block - 1) / block) * block;

    // if ((unsigned long int)N > FMATMUL_MAX_N ||
    //     (unsigned long int)K > FMATMUL_MAX_K ||
    //     padded_m > FMATMUL_MAX_M)
    // {
    //     matrix_multiply_scalar_nt(a, b, c, M, N, K);
    //     return;
    // }

    if (padded_m == (unsigned long int)M) {
        // No row padding: write straight into c.
        fmatmul_nt_32(c, a, b,
                   (unsigned long int)M, (unsigned long int)N, (unsigned long int)K);
        return;
    }

    const size_t mn = (size_t)M * (size_t)N;
    const size_t pnk = (size_t)padded_m * (size_t)N;
    const size_t mk = (size_t)M * (size_t)K;

    // Pack a into the padded scratch (rows [M, padded_m) zero-filled).
    size_t remaining_mn = mn;
    const float *src_mn = a;
    float *dst_mn = fmatmul_a_scratch;
    while (remaining_mn > 0)
    {
        size_t vl = 0;
        asm volatile("vsetvli %0, %1, e32, m1, ta, ma" : "=r"(vl) : "r"(remaining_mn));
        asm volatile("vle32.v v0, (%0);" : : "r"(src_mn) : "memory");
        asm volatile("vse32.v v0, (%0);" : : "r"(dst_mn) : "memory");
        src_mn += vl;
        dst_mn += vl;
        remaining_mn -= vl;
    }

    size_t remaining = pnk - mn;
    float *dst = fmatmul_a_scratch + mn;
    while (remaining > 0)
    {
        size_t vl = 0;
        asm volatile("vsetvli %0, %1, e32, m1, ta, ma" : "=r"(vl) : "r"(remaining));
        asm volatile("vmv.v.x v0, zero");
        asm volatile("vse32.v v0, (%0);" : : "r"(dst) : "memory");
        dst += vl;
        remaining -= vl;
    }

    // fmatmul writes padded_m rows, so it cannot target c directly here.
    fmatmul_nt_32(fmatmul_c_scratch_32, fmatmul_a_scratch, b,
               padded_m, (unsigned long int)N, (unsigned long int)K);

    // Copy the valid M*K rows into c.
    size_t remaining_mk = mk;
    const float *src_mk = fmatmul_c_scratch_32;
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

void matrix_multiply_tn_32(const float *a, const float *b, float *c,
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
        fmatmul_tn_32(c, a, b,
                   (unsigned long int)M, (unsigned long int)N, (unsigned long int)K);
        return;
    }

    if (((size_t)M * (size_t)padded_n) > MATRIX_TRANSPOSE_WORKSPACE_ELEMS) {
        printf_("[SCALAR] matrix_multiply_tn_32: M=%d N=%d K=%d (workspace too small)\n", M, N, K);
        matrix_multiply_scalar_tn(a, b, c, M, N, K);
        return;
    }

    float *a_pad = shared_memory_pool_32;
    for (int m = 0; m < M; m++)
    {
        const float *src_row = a + (size_t)m * (size_t)N;
        float *dst_row = a_pad + (size_t)m * (size_t)padded_n;

        size_t remaining_row = (size_t)N;
        while (remaining_row > 0)
        {
            size_t vl = 0;
            asm volatile("vsetvli %0, %1, e32, m1, ta, ma" : "=r"(vl) : "r"(remaining_row));
            asm volatile("vle32.v v0, (%0);" : : "r"(src_row) : "memory");
            asm volatile("vse32.v v0, (%0);" : : "r"(dst_row) : "memory");
            src_row += vl;
            dst_row += vl;
            remaining_row -= vl;
        }

        size_t remaining_pad = (size_t)padded_n - (size_t)N;
        while (remaining_pad > 0)
        {
            size_t vl = 0;
            asm volatile("vsetvli %0, %1, e32, m1, ta, ma" : "=r"(vl) : "r"(remaining_pad));
            asm volatile("vmv.v.x v0, zero");
            asm volatile("vse32.v v0, (%0);" : : "r"(dst_row) : "memory");
            dst_row += vl;
            remaining_pad -= vl;
        }
    }

    fmatmul_tn_32(c, a_pad, b,
               (unsigned long int)M, (unsigned long int)N, (unsigned long int)K);
}

int matrix_multiply_tn_verify_32(const float *a, const float *b,
                              const int M, const int N, const int K,
                              const float eps)
{
    if (M <= 0 || N <= 0 || K <= 0)
        return 0;

    const size_t nk = (size_t)N * (size_t)K;
    if (nk > MATRIX_TRANSPOSE_WORKSPACE_ELEMS) {
        printf_("matrix_multiply_tn_verify_32: dims too large (%d x %d)\n", N, K);
        return 0;
    }

    float *tn_out = shared_memory_pool_32;
    for (size_t idx = 0; idx < nk; idx++)
        tn_out[idx] = 0.0f;

    fmatmul_tn_32(tn_out, a, b,
               (unsigned long int)M, (unsigned long int)N, (unsigned long int)K);

    size_t mismatch_count = 0;
    float max_diff = 0.0f;

    for (int n = 0; n < N; n++)
    {
        for (int k = 0; k < K; k++)
        {
            float sum = 0.0f;
            for (int m = 0; m < M; m++)
                sum += a[(size_t)m * (size_t)N + (size_t)n] *
                       b[(size_t)m * (size_t)K + (size_t)k];

            float diff = fabsf(sum - tn_out[(size_t)n * (size_t)K + (size_t)k]);
            if (diff > max_diff)
                max_diff = diff;
            if (diff > eps)
                mismatch_count++;
        }
    }

    if (mismatch_count == 0)
        printf_("matrix_multiply_tn_verify_32: OK (max diff %f)\n", max_diff);
    else
        printf_("matrix_multiply_tn_verify_32: FAIL (mismatches %lu, max diff %f)\n",
                (unsigned long)mismatch_count, max_diff);

    return (mismatch_count == 0) ? 1 : 0;
}

void matrix_multiply_nt_deferred_32(const float *a, const float *b, float *c,
                                 const int M, const int N, const int K)
{
    /**
     * matrix multiply, c += a * b^T (deferred reduction, 4x4 only)
     *
     * Input:
     * a    [M,N]
     * b    [K,N]
     * Output:
     * c    [M,K]
     * */
    if (M <= 0 || N <= 0 || K <= 0)
        return;

    if (((unsigned long int)K % 4) != 0) {
        matrix_multiply_nt_32(a, b, c, M, N, K);
        return;
    }

    float *fmatmul_a_scratch = shared_memory_pool_32;
    unsigned long int padded_m = (((unsigned long int)M + 3) / 4) * 4;

    // if ((unsigned long int)N > FMATMUL_MAX_N ||
    //     (unsigned long int)K > FMATMUL_MAX_K ||
    //     padded_m > FMATMUL_MAX_M)
    // {
    //     matrix_multiply_nt_32(a, b, c, M, N, K);
    //     return;
    // }

    const size_t mn = (size_t)M * (size_t)N;
    const size_t pnk = (size_t)padded_m * (size_t)N;
    const size_t mk = (size_t)M * (size_t)K;
    const size_t mk_pad = (size_t)padded_m * (size_t)K;

    for (size_t idx = 0; idx < mn; idx++)
        fmatmul_a_scratch[idx] = a[idx];
    for (size_t idx = mn; idx < pnk; idx++)
        fmatmul_a_scratch[idx] = 0.0f;
    for (size_t idx = 0; idx < mk_pad; idx++)
        fmatmul_c_scratch_32[idx] = 0.0f;

    fmatmul_4x4_deferred_32(fmatmul_c_scratch_32, fmatmul_a_scratch, b,
                         padded_m, (unsigned long int)N, (unsigned long int)K);

    for (size_t idx = 0; idx < mk; idx++)
        c[idx] += fmatmul_c_scratch_32[idx];
}

int matrix_multiply_nt_verify_32(const float *a, const float *b,
                              const int M, const int N, const int K,
                              const float eps)
{
    if (M <= 0 || N <= 0 || K <= 0)
        return 0;

    // if ((unsigned long int)M > FMATMUL_MAX_M ||
    //     (unsigned long int)K > FMATMUL_MAX_K)
    // {
    //     printf_("matrix_multiply_nt_verify_32: dims too large (%d x %d)\n", M, K);
    //     return 0;
    // }

    const size_t mk = (size_t)M * (size_t)K;
    for (size_t idx = 0; idx < mk; idx++) {
        fmatmul_nt_loop_scratch[idx] = 0.0f;
        fmatmul_nt_out_scratch[idx] = 0.0f;
    }

    matrix_multiply_scalar_nt(a, b, fmatmul_nt_loop_scratch, M, N, K);
    matrix_multiply_nt_32(a, b, fmatmul_nt_out_scratch, M, N, K);

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
        printf_("matrix_multiply_nt_verify_32: OK (max diff %f)\n", max_diff);
    else
        printf_("matrix_multiply_nt_verify_32: FAIL (mismatches %lu, max diff %f)\n",
                (unsigned long)mismatch_count, max_diff);

    return (mismatch_count == 0) ? 1 : 0;
}

void matrix_transpose_32(float *x, int m, int n)
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
        printf_("Error: matrix_transpose_32 workspace too small for %d x %d\n", m, n);
        exit(1);
    }
    float *tmp = shared_memory_pool_32;
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



static void matrix_multiply_scalar(const float *a, const float *b, float *c, const int M, const int N, const int K)
{
    register int i, j, p;
    register const float *a_ptr = a;
    for (i = 0; i < M; i++)
    {
        register const float *b_ptr = b;
        for (j = 0; j < N; j++)
        {
            register float apart = *(a_ptr++);
            if (apart < 0.00001f && apart > -0.00001f)
            {
                b_ptr += K;
                continue;
            }
            register float *c_ptr = c + i * K;
            for (p = 0; p < K; p++)
                *(c_ptr++) += *(b_ptr++) * apart;
        }
    }
}

static void matrix_multiply_scalar_nt(const float *a, const float *b, float *c,
                                      const int M, const int N, const int K)
{
    for (int i = 0; i < M; i++)
    {
        const float *a_ptr = a + (size_t)i * (size_t)N;
        for (int k = 0; k < K; k++)
        {
            const float *b_ptr = b + (size_t)k * (size_t)N;
            float sum = 0.0f;
            for (int j = 0; j < N; j++)
                sum += a_ptr[j] * b_ptr[j];
            c[(size_t)i * (size_t)K + (size_t)k] += sum;
        }
    }
}

static void matrix_multiply_scalar_tn(const float *a, const float *b, float *c,
                                      const int M, const int N, const int K)
{
    for (int n = 0; n < N; n++)
    {
        for (int k = 0; k < K; k++)
        {
            float sum = 0.0f;
            for (int m = 0; m < M; m++)
                sum += a[(size_t)m * (size_t)N + (size_t)n] *
                       b[(size_t)m * (size_t)K + (size_t)k];
            c[(size_t)n * (size_t)K + (size_t)k] += sum;
        }
    }
}

static void matrix_multiply_scalar_fused(const float *a, const float *b,
                                         const float *bias, float *c,
                                         const int M, const int N, const int K)
{
    register int i, j, p;
    register const float *a_ptr = a;
    for (i = 0; i < M; i++)
    {
        float *c_ptr = c + i * K;
        const float *bias_ptr = bias;
        for (p = 0; p < K; p++)
            *(c_ptr++) = *(bias_ptr++);
    }
    for (i = 0; i < M; i++)
    {
        register const float *b_ptr = b;
        for (j = 0; j < N; j++)
        {
            register float apart = *(a_ptr++);
            if (apart < 0.00001f && apart > -0.00001f)
            {
                b_ptr += K;
                continue;
            }
            register float *c_ptr = c + i * K;
            for (p = 0; p < K; p++)
                *(c_ptr++) += *(b_ptr++) * apart;
        }
    }
}