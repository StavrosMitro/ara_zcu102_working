//
// File:        matrix.c
// Description: Pure FP16 matrix computation
// Author:      Haris Wang
// FP16 refactor: Stavros Mitropoulos, NTUA
//
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "alexnet.h"
#include "fmatmul.h"
#ifdef SPIKE
#include <printf.h>
#elif defined ARA_LINUX
#include <stdio.h>
#else
#include "printf.h"
#endif

#define MATRIX_TRANSPOSE_WORKSPACE_ELEMS (FC_MAX_IN_UNITS * FC_MAX_INTERNAL)

_Float16 shared_memory_pool_16[MATRIX_TRANSPOSE_WORKSPACE_ELEMS];

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

_Float16 fmatmul_c_scratch_16[FMATMUL_MAX_M * FMATMUL_MAX_K];
static _Float16 fmatmul_nt_loop_scratch[FMATMUL_MAX_M * FMATMUL_MAX_K];
static _Float16 fmatmul_nt_out_scratch[FMATMUL_MAX_M * FMATMUL_MAX_K];

static void matrix_multiply_scalar(const _Float16 *a, const _Float16 *b, _Float16 *c,
                                   const int M, const int N, const int K);
static void matrix_multiply_scalar_nt(const _Float16 *a, const _Float16 *b, _Float16 *c,
                                      const int M, const int N, const int K);
static void matrix_multiply_scalar_tn(const _Float16 *a, const _Float16 *b, _Float16 *c,
                                      const int M, const int N, const int K);
static void matrix_multiply_scalar_fused(const _Float16 *a, const _Float16 *b,
                                         const _Float16 *bias, _Float16 *c,
                                         const int M, const int N, const int K);

static inline unsigned long int fmatmul_row_block(unsigned long int m)
{
    if (m <= 4)   return 4;
    if (m <= 8)   return 8;
    if (m <= 64)  return 16;
    if (m <= 128) return 8;
    return 4;
}

void matrix_multiply_16(const _Float16 *a, const _Float16 *b, _Float16 *c,
                        const int M, const int N, const int K)
{
    if (M <= 0 || N <= 0 || K <= 0)
        return;
    _Float16 *fmatmul_a_scratch = shared_memory_pool_16;
    unsigned long int block = fmatmul_row_block((unsigned long int)M);
    unsigned long int padded_m = (((unsigned long int)M + block - 1) / block) * block;

    if ((unsigned long int)N > FMATMUL_MAX_N ||
        (unsigned long int)K > FMATMUL_MAX_K ||
        padded_m > FMATMUL_MAX_M)
    {
        printf_("[SCALAR] matrix_multiply_16: M=%d N=%d K=%d padded_m=%lu MAX_M=%d\n", M, N, K, padded_m, FMATMUL_MAX_M);
        // matrix_multiply_scalar() accumulates (`*c_ptr += ...`), so zero c
        // first to keep this path's contract identical to the vector paths.
        memset(c, 0, (size_t)M * (size_t)K * sizeof(_Float16));
        matrix_multiply_scalar(a, b, c, M, N, K);
        return;
    }

    if (padded_m == (unsigned long int)M) {
        fmatmul_16(c, a, b,
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

    fmatmul_16(fmatmul_c_scratch_16, fmatmul_a_scratch, b,
            padded_m, (unsigned long int)N, (unsigned long int)K);

    size_t remaining_mk = mk;
    const _Float16 *src_mk = fmatmul_c_scratch_16;
    _Float16 *dst_mk = c;
    while (remaining_mk > 0)
    {
        size_t vl = 0;
        asm volatile("vsetvli %0, %1, e16, m1, ta, ma" : "=r"(vl) : "r"(remaining_mk));
        // Plain copy-back: the result in fmatmul_c_scratch_16 IS the answer.
        // This used to vle16 the old c and vfadd into it, ACCUMULATING --
        // which disagreed with fmatmul_16() overwriting c on the unpadded
        // path, and (for the _nt variant) with matrix_multiply_nt_32, whose
        // equivalent already overwrites. Callers that did not pre-zero c
        // silently summed every invocation onto the previous one.
        asm volatile("vle16.v v0, (%0);" : : "r"(src_mk) : "memory");
        asm volatile("vse16.v v0, (%0);" : : "r"(dst_mk) : "memory");
        src_mk += vl;
        dst_mk += vl;
        remaining_mk -= vl;
    }
}

void matrix_multiply_fused_16(const _Float16 *a, const _Float16 *b, const _Float16 *bias,
                               _Float16 *c, const int M, const int N, const int K)
{
    if (M <= 0 || N <= 0 || K <= 0)
        return;

    _Float16 *fmatmul_a_scratch = shared_memory_pool_16;
    unsigned long int block = fmatmul_row_block((unsigned long int)M);
    unsigned long int padded_m = (((unsigned long int)M + block - 1) / block) * block;

    if ((unsigned long int)N > FMATMUL_MAX_N ||
        (unsigned long int)K > FMATMUL_MAX_K ||
        padded_m > FMATMUL_MAX_M)
    {
        printf_("[SCALAR] matrix_multiply_fused_16: M=%d N=%d K=%d padded_m=%lu MAX_M=%d\n", M, N, K, padded_m, FMATMUL_MAX_M);
        matrix_multiply_scalar_fused(a, b, bias, c, M, N, K);
        return;
    }

    if (padded_m == (unsigned long int)M) {
        fmatmul_fused_16(c, a, b, bias,
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

    fmatmul_fused_16(fmatmul_c_scratch_16, fmatmul_a_scratch, b, bias,
                 padded_m, (unsigned long int)N, (unsigned long int)K);

    size_t remaining_mk = mk;
    const _Float16 *src_mk = fmatmul_c_scratch_16;
    _Float16 *dst_mk = c;
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

void matrix_multiply_nt_16(const _Float16 *a, const _Float16 *b, _Float16 *c,
                            const int M, const int N, const int K)
{
    if (M <= 0 || N <= 0 || K <= 0)
        return;

    _Float16 *fmatmul_a_scratch = shared_memory_pool_16;
    unsigned long int block = fmatmul_row_block((unsigned long int)M);
    unsigned long int padded_m = (((unsigned long int)M + block - 1) / block) * block;

    const size_t mn = (size_t)M * (size_t)N;
    const size_t pnk = (size_t)padded_m * (size_t)N;
    const size_t mk = (size_t)M * (size_t)K;

    if (padded_m == (unsigned long int)M) {
        // No padding: write straight into c and skip the copy-back entirely.
        // Routing through fmatmul_c_scratch_16 and then copying M*K elements
        // back cost this app a load+store pass that matrix_multiply_nt_32
        // (which already short-circuits here) never paid -- pure asymmetric
        // memory traffic in the FP16 vs FP32 comparison, measurable as a
        // SLOWER backward_d_input for FP16 despite the narrower elements.
        fmatmul_nt_16(c, a, b,
                   (unsigned long int)M, (unsigned long int)N, (unsigned long int)K);
        return;
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

        fmatmul_nt_16(fmatmul_c_scratch_16, fmatmul_a_scratch, b,
                   padded_m, (unsigned long int)N, (unsigned long int)K);
    }

    size_t remaining_mk = mk;
    const _Float16 *src_mk = fmatmul_c_scratch_16;
    _Float16 *dst_mk = c;
    while (remaining_mk > 0)
    {
        size_t vl = 0;
        asm volatile("vsetvli %0, %1, e16, m1, ta, ma" : "=r"(vl) : "r"(remaining_mk));
        // Plain copy-back: the result in fmatmul_c_scratch_16 IS the answer.
        // This used to vle16 the old c and vfadd into it, ACCUMULATING --
        // which disagreed with fmatmul_16() overwriting c on the unpadded
        // path, and (for the _nt variant) with matrix_multiply_nt_32, whose
        // equivalent already overwrites. Callers that did not pre-zero c
        // silently summed every invocation onto the previous one.
        asm volatile("vle16.v v0, (%0);" : : "r"(src_mk) : "memory");
        asm volatile("vse16.v v0, (%0);" : : "r"(dst_mk) : "memory");
        src_mk += vl;
        dst_mk += vl;
        remaining_mk -= vl;
    }
}

void matrix_multiply_tn_16(const _Float16 *a, const _Float16 *b, _Float16 *c,
                            const int M, const int N, const int K)
{
    if (M <= 0 || N <= 0 || K <= 0)
        return;

    unsigned long int block = fmatmul_row_block((unsigned long int)N);
    unsigned long int padded_n = (((unsigned long int)N + block - 1) / block) * block;

    if (padded_n == (unsigned long int)N) {
        fmatmul_tn_16(c, a, b,
                   (unsigned long int)M, (unsigned long int)N, (unsigned long int)K);
        return;
    }

    if (((size_t)M * (size_t)padded_n) > MATRIX_TRANSPOSE_WORKSPACE_ELEMS) {
        printf_("[SCALAR] matrix_multiply_tn_16: M=%d N=%d K=%d (workspace too small)\n", M, N, K);
        matrix_multiply_scalar_tn(a, b, c, M, N, K);
        return;
    }

    _Float16 *a_pad = shared_memory_pool_16;
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

    fmatmul_tn_16(c, a_pad, b,
               (unsigned long int)M, (unsigned long int)N, (unsigned long int)K);
}

int matrix_multiply_tn_verify_16(const _Float16 *a, const _Float16 *b,
                                  const int M, const int N, const int K,
                                  const _Float16 eps)
{
    if (M <= 0 || N <= 0 || K <= 0)
        return 0;

    const size_t nk = (size_t)N * (size_t)K;
    if (nk > MATRIX_TRANSPOSE_WORKSPACE_ELEMS) {
        printf_("matrix_multiply_tn_verify_16: dims too large (%d x %d)\n", N, K);
        return 0;
    }

    _Float16 *tn_out = shared_memory_pool_16;
    for (size_t idx = 0; idx < nk; idx++)
        tn_out[idx] = (_Float16)0.0f;

    fmatmul_tn_16(tn_out, a, b,
               (unsigned long int)M, (unsigned long int)N, (unsigned long int)K);

    size_t mismatch_count = 0;
    _Float16 max_diff = (_Float16)0.0f;
    const _Float16 inv_batch = (_Float16)(1.0f / (float)M);

    for (int n = 0; n < N; n++)
    {
        for (int k = 0; k < K; k++)
        {
            _Float16 sum = (_Float16)0.0f;
            for (int m = 0; m < M; m++)
                sum += a[(size_t)m * (size_t)N + (size_t)n] *
                       b[(size_t)m * (size_t)K + (size_t)k];
            sum *= inv_batch;

            _Float16 diff = sum - tn_out[(size_t)n * (size_t)K + (size_t)k];
            if (diff < (_Float16)0.0f) diff = -diff;
            if (diff > max_diff) max_diff = diff;
            if (diff > eps) mismatch_count++;
        }
    }

    if (mismatch_count == 0)
        printf_("matrix_multiply_tn_verify_16: OK\n");
    else
        printf_("matrix_multiply_tn_verify_16: FAIL (mismatches %lu)\n",
                (unsigned long)mismatch_count);

    return (mismatch_count == 0) ? 1 : 0;
}

void matrix_multiply_nt_deferred_16(const _Float16 *a, const _Float16 *b, _Float16 *c,
                                    const int M, const int N, const int K)
{
    if (M <= 0 || N <= 0 || K <= 0)
        return;

    if (((unsigned long int)K % 4) != 0) {
        matrix_multiply_nt_16(a, b, c, M, N, K);
        return;
    }

    _Float16 *fmatmul_a_scratch = shared_memory_pool_16;
    unsigned long int padded_m = (((unsigned long int)M + 3) / 4) * 4;

    const size_t mn = (size_t)M * (size_t)N;
    const size_t pnk = (size_t)padded_m * (size_t)N;
    const size_t mk = (size_t)M * (size_t)K;
    const size_t mk_pad = (size_t)padded_m * (size_t)K;

    for (size_t idx = 0; idx < mn;     idx++) fmatmul_a_scratch[idx]    = a[idx];
    for (size_t idx = mn; idx < pnk;   idx++) fmatmul_a_scratch[idx]    = (_Float16)0.0f;
    for (size_t idx = 0; idx < mk_pad; idx++) fmatmul_c_scratch_16[idx] = (_Float16)0.0f;

    fmatmul_4x4_deferred_16(fmatmul_c_scratch_16, fmatmul_a_scratch, b,
                         padded_m, (unsigned long int)N, (unsigned long int)K);

    for (size_t idx = 0; idx < mk; idx++)
        c[idx] += fmatmul_c_scratch_16[idx];
}

int matrix_multiply_nt_verify_16(const _Float16 *a, const _Float16 *b,
                                  const int M, const int N, const int K,
                                  const _Float16 eps)
{
    if (M <= 0 || N <= 0 || K <= 0)
        return 0;

    const size_t mk = (size_t)N * (size_t)K;
    for (size_t idx = 0; idx < mk; idx++) {
        fmatmul_nt_loop_scratch[idx] = (_Float16)0.0f;
        fmatmul_nt_out_scratch[idx]  = (_Float16)0.0f;
    }

    matrix_multiply_scalar_nt(a, b, fmatmul_nt_loop_scratch, M, N, K);
    matrix_multiply_nt_16(a, b, fmatmul_nt_out_scratch, M, N, K);

    size_t mismatch_count = 0;
    _Float16 max_diff = (_Float16)0.0f;
    for (size_t idx = 0; idx < mk; idx++) {
        _Float16 diff = fmatmul_nt_loop_scratch[idx] - fmatmul_nt_out_scratch[idx];
        if (diff < (_Float16)0.0f) diff = -diff;
        if (diff > max_diff) max_diff = diff;
        if (diff > eps) mismatch_count++;
    }

    if (mismatch_count == 0)
        printf_("matrix_multiply_nt_verify_16: OK\n");
    else
        printf_("matrix_multiply_nt_verify_16: FAIL (mismatches %lu)\n",
                (unsigned long)mismatch_count);

    return (mismatch_count == 0) ? 1 : 0;
}

void matrix_transpose_16(_Float16 *x, int m, int n)
{
    size_t elems = (size_t)m * (size_t)n;
    if (elems > MATRIX_TRANSPOSE_WORKSPACE_ELEMS) {
        printf_("Error: matrix_transpose_16 workspace too small for %d x %d\n", m, n);
        exit(1);
    }
    _Float16 *tmp = shared_memory_pool_16;
    register int i, j;
    register _Float16 *ptr = x;
    for (i = 0; i < m; i++)
    {
        for (j = 0; j < n; j++)
            tmp[j*m+i] = *(ptr++);
    }
    memcpy(x, tmp, elems * sizeof(_Float16));
}

static void matrix_multiply_scalar(const _Float16 *a, const _Float16 *b, _Float16 *c,
                                   const int M, const int N, const int K)
{
    register int i, j, p;
    register const _Float16 *a_ptr = a;
    for (i = 0; i < M; i++)
    {
        register const _Float16 *b_ptr = b;
        for (j = 0; j < N; j++)
        {
            register _Float16 apart = *(a_ptr++);
            if (apart == (_Float16)0.0f)
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

static void matrix_multiply_scalar_nt(const _Float16 *a, const _Float16 *b, _Float16 *c,
                                      const int M, const int N, const int K)
{
    for (int i = 0; i < M; i++)
    {
        const _Float16 *a_ptr = a + (size_t)i * (size_t)N;
        for (int k = 0; k < K; k++)
        {
            const _Float16 *b_ptr = b + (size_t)k * (size_t)N;
            _Float16 sum = (_Float16)0.0f;
            for (int j = 0; j < N; j++)
                sum += a_ptr[j] * b_ptr[j];
            c[(size_t)i * (size_t)K + (size_t)k] += sum;
        }
    }
}

static void matrix_multiply_scalar_tn(const _Float16 *a, const _Float16 *b, _Float16 *c,
                                      const int M, const int N, const int K)
{
    for (int n = 0; n < N; n++)
    {
        for (int k = 0; k < K; k++)
        {
            _Float16 sum = (_Float16)0.0f;
            for (int m = 0; m < M; m++)
                sum += a[(size_t)m * (size_t)N + (size_t)n] *
                       b[(size_t)m * (size_t)K + (size_t)k];
            c[(size_t)n * (size_t)K + (size_t)k] += sum;
        }
    }
}

static void matrix_multiply_scalar_fused(const _Float16 *a, const _Float16 *b,
                                         const _Float16 *bias, _Float16 *c,
                                         const int M, const int N, const int K)
{
    for (int i = 0; i < M; i++) {
        for (int p = 0; p < K; p++) {
            _Float16 acc = bias[p];
            for (int j = 0; j < N; j++)
                acc += a[i * N + j] * b[j * K + p];
            c[i * K + p] = acc;
        }
    }
}
