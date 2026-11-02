//
// File:        dropout_layer.c
// Description: inverted dropout with a stored mask for the backward pass
// Author:      Haris Wang
//
#include <stdlib.h>
#include <math.h>
#include "dropout_layer.h"
#ifdef SPIKE
#include <printf.h>
#elif defined ARA_LINUX
#include <stdio.h>
#else
#include "printf.h"
#endif

void dropout_forward(float *x, float *mask, float prob, int n)
{
    if (prob <= 0.0f) {
        // No dropout: identity, and a mask of ones so backward is a no-op.
        for (int i = 0; i < n; i++) mask[i] = 1.0f;
        return;
    }

    float keep  = 1.0f - prob;
    float scale = 1.0f / keep;
    for (int i = 0; i < n; i++) {
        float u = (float)rand() / ((float)RAND_MAX + 1.0f);   // [0, 1)
        if (u < keep) {
            mask[i] = scale;
            x[i]   *= scale;
        } else {
            mask[i] = 0.0f;
            x[i]    = 0.0f;
        }
    }
}

void dropout_backward(float *dx, const float *mask, int n)
{
    for (int i = 0; i < n; i++)
        dx[i] *= mask[i];
}
