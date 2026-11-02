//
// File:        matrix.c
// Description: Implementation of matrix computation
// Author:      Haris Wang
//
#include <stdlib.h>
#include <string.h>
#include "alexnet.h"
// #include <immintrin.h> 
#ifdef SPIKE
#include <printf.h>
#elif defined ARA_LINUX
#include <stdio.h>
#else
#include "printf.h"
#endif

// I REMOVED IT, BECAUSE IT IS FOR X86 ASSEMBLY

// Workspace used by matrix_transpose to avoid large stack allocations.
// Sized conservatively for current transpose use cases.
#define MATRIX_TRANSPOSE_WORKSPACE_ELEMS 2000000
static float matrix_transpose_workspace[MATRIX_TRANSPOSE_WORKSPACE_ELEMS];

void matrix_multiply_eps(const float *a, const float *b, float *c, const int M, const int N, const int K, const float eps)
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
    // Each output row i is written only by iteration i (c + i*K are disjoint
    // spans) and a/b are read-only, so the i-loop carries no dependency and is
    // parallelised across cores. The `if` clause keeps tiny GEMMs serial so the
    // fork/join cost never outweighs the work. Without -fopenmp this pragma is
    // ignored and the loop runs exactly as before.
    #pragma omp parallel for schedule(static) if((long)M * N * K > 100000)
    for (int i = 0; i < M; i++)
    {
        const float *a_row = a + (size_t)i * N;
        float *c_row = c + (size_t)i * K;
        const float *b_ptr = b;
        for (int j = 0; j < N; j++)
        {
            float apart = a_row[j];
            if (apart < eps && apart > -eps) //masking for vector processing
            {
                b_ptr += K;
                continue;
            }
            for (int p = 0; p < K; p++)
                c_row[p] += b_ptr[p] * apart;
            b_ptr += K;
        }
    }
}

void matrix_multiply(const float *a, const float *b, float *c, const int M, const int N, const int K)
{
    matrix_multiply_eps(a, b, c, M, N, K, 0.00001f);
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
    float *tmp = matrix_transpose_workspace;
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
