//
// File:        inference.c
// Description: Implementation of inference function
// Author:      Haris Wang
//
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <time.h>
#include "alexnet.h"
#include "data.h"
#include "image_inference.h"
#ifdef SPIKE
#include <printf.h>
#elif defined ARA_LINUX
#include <stdio.h>
#else
#include "printf.h"
#endif


// Staging buffer: the image pipeline is FP32, but the net input is FP16.
static _Float16 infer_input_buf[IN_CHANNELS * FEATURE0_L * FEATURE0_L];

void alexnet_inference(alexnet *net, const unsigned char *img_bytes)
{
    image img;
    const unsigned char *src = (img_bytes != NULL) ? img_bytes : img_data;
    img = load_image(src, FEATURE0_L, FEATURE0_L, IN_CHANNELS, 0);
    const int n_in = IN_CHANNELS * FEATURE0_L * FEATURE0_L;
    for (int i = 0; i < n_in; i++) infer_input_buf[i] = (_Float16)img.data[i];  // narrow to FP16
    net->input = infer_input_buf;
    forward_alexnet(net);
    int pred = argmax(net->output, OUT_LAYER);
    printf_("prediction: %d\n", pred);
    free_forward_activations(net);
    free_image(&img);
}
