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


static _Float16 inference_input_buf_16[FEATURE0_L * FEATURE0_L * IN_CHANNELS];

void alexnet_inference_16(alexnet *net, const unsigned char *img_bytes)
{
    image img;
    const unsigned char *src = (img_bytes != NULL) ? img_bytes : img_data_32;
    img = load_image_32(src, FEATURE0_L, FEATURE0_L, IN_CHANNELS, 0);
    for (int i = 0; i < FEATURE0_L * FEATURE0_L * IN_CHANNELS; i++)
        inference_input_buf_16[i] = (_Float16)img.data[i];
    net->input = inference_input_buf_16;
    forward_alexnet(net);
    int pred = argmax(net->output, OUT_LAYER);
    printf_("prediction: %d\n", pred);
    // free_forward_activations(net);
    // free_image_32(&img);
}
