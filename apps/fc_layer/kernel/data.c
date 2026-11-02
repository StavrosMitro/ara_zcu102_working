//
// File:        data.c
// Description: Provide functions for data process 
// Author:      Haris Wang
// Modified and corrected: Stavros Mitropoulos
//
// #include <stdlib.h>
#include <stdio.h>
// #include <assert.h>
#include <string.h>
#include "data.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "weights.h"
#include "image_inference.h"
#include "cifar10_dataset.h"
#ifdef SPIKE
#include <printf.h>
#elif defined ARA_LINUX
#include <stdio.h>
#else
#include "printf.h"
#endif


#define CIFAR10_W 32
#define CIFAR10_H 32
#define CIFAR10_C 3
#define IMAGE_SCRATCH_SLOTS 8
#define IMAGE_MAX_FLOATS (CIFAR10_W * CIFAR10_H * CIFAR10_C)
#define FC_INPUT_UNITS 2048
#define FC_DATASET_COUNT 4

static float image_scratch[IMAGE_SCRATCH_SLOTS][IMAGE_MAX_FLOATS];
static int image_scratch_cursor = 0;

static int dataset_cursor = 0;


void make_image(image *img, int w, int h, int c)
{
    /**
     * Make image
     * 
     * Input:
     *      w, h, c
     * Output:
     *      img
     * */
    // printf_("entering make_image\n");
    img->w = w;
    img->h = h;
    img->c = c;
    if (w * h * c > IMAGE_MAX_FLOATS) {
        printf_("Error! make_image requested %d floats, max supported is %d\n", w * h * c, IMAGE_MAX_FLOATS);
        exit(1);
    }
    img->data = image_scratch[image_scratch_cursor];
    image_scratch_cursor = (image_scratch_cursor + 1) % IMAGE_SCRATCH_SLOTS;
    // printf_("malloc completed, no segmetation fault in make_image function\n");
}

void free_image(image *img)
{
    // Static-buffer mode: no heap ownership to release.
    img->data = NULL;
}


static float get_pixel(image *img, unsigned int x, unsigned int y, unsigned int c)
{
    // assert(x<(img->w) && y<(img->h) && c<(img->c));
    return img->data[c*img->w*img->h+y*img->w+x];
}

static void set_pixel(image *img, float value, unsigned int x, unsigned int y, unsigned int c)
{
    if (x < 0 || y < 0 || c < 0 || x >= (img->w) || y >= (img->h) || c >= (img->c)) return;
    // assert(x<(img->w) && y<(img->h) && c<(img->c));
    img->data[c*(img->w)*(img->h)+y*(img->w)+x] = value;
}

static void add_pixel(image *img, float value, unsigned int x, unsigned int y, unsigned int c)
{
    // assert(x<(img->w) && y<(img->h) && c<(img->c));
    img->data[c*(img->w)*(img->h)+y*(img->w)+x] += value;
}

void horizontal_flip(image *im)
{
    /**
     * Data argumention : horizontal flip
     * */
    image tmp;
    make_image(&tmp, im->w, im->h, im->c);
    for (short z = 0; z < im->c; z++)
    {
        for (short y = 0; y < im->h; y++)
        {
            register int st_idx = y * im->w + z * im->w * im->h;
            for (short x = 0; x < im->w; x++)
                tmp.data[st_idx + x] = im->data[st_idx + (im->w - 1 - x)];
        }
    }
    memcpy(im->data, tmp.data, im->w * im->h * im->c * sizeof(float));
}

void resize_image(image *im, int w, int h)
{
    //here im->h and im->c express the original dims of the image, 
    //while through passing values w,h are the wanted by the network
    image resized, part;
    make_image(&resized, w, h, im->c);
    make_image(&part,  w, im->h, im->c);
    float   w_scale = (im->w-1) * 1.0 / (w-1),
            h_scale = (im->h-1) * 1.0 / (h-1);
    register float val;
    register unsigned int r, c, k;
    for (k = 0; k < im->c; ++k){
        for (r = 0; r < im->h; ++r){
            for (c = 0; c < w; ++c){
                val = 0;
                if (c == w-1 || im->w == 1) {
                    val = get_pixel(im, im->w-1, r, k);
                }else {
                    float sx = c * w_scale;
                    int   ix = (int)sx;
                    float dx = sx - ix;
                    val = (1 - dx) * get_pixel(im, ix, r, k) + dx * get_pixel(im, ix+1, r, k);
                }
                set_pixel(&part, val, c, r, k);
            }
        }
    }
    for (k = 0; k < im->c; ++k){
        for (r = 0; r < h; ++r){
            float sy = r * h_scale;
            int   iy = (int) sy;
            float dy = sy - iy;
            for (c = 0; c < w; ++c){
                val = (1-dy) * get_pixel(&part, c, iy, k);
                set_pixel(&resized, val, c, r, k);
            }
            if (r == h-1 || im->h == 1) continue;
            for (c = 0; c < w; ++c){
                val = dy * get_pixel(&part, c, iy+1, k);
                add_pixel(&resized, val, c, r, k);
            }
        }
    }

    // copy resized pixels back into the original image storage
    memcpy(im->data, resized.data, (size_t)(w * h * im->c) * sizeof(float));
    im->w = resized.w;
    im->h = resized.h;
    im->c = resized.c;
}

image load_image(const unsigned char *start_img, int W, int H, int channels, int is_h_flip) //we want to pass a pointer to the start of the array of each image.
{ //so that 4d array w,h,c,image and we traverse through arrays




    /**
     * load image from file
     * 
     * Input:
     *      filename
     *      channels
     *      is_h_filp   whether to apply horizontal flip
     * Return:
     *      image
     * */

    int w=img_data_w;
    int h=img_data_h;
    int c=img_data_c;
    const unsigned char *data = start_img;
    if (!data)
    {
        printf_("Error! Can't load image %p! \n", start_img);
        image empty;
        int fallback_c = channels ? channels : img_data_c;
        make_image(&empty, W, H, fallback_c);
        memset(empty.data, 0, W * H * fallback_c * sizeof(float));
        return empty;
    }
    if (channels)
    {
        c=channels;
    } 
    image img;
    make_image(&img, w, h, c);
    register int dst_idx, src_idx;
    for (int k = 0; k < c; k++)
    {
        for (int j = 0; j < h; j++)
        {
            for (int i = 0; i < w; i++)
            {
                dst_idx = i + w*j + w*h*k;
                src_idx = k + c*i + c*w*j;
                img.data[dst_idx] = (float)data[src_idx] / 127.5 - 1;
            }
        }
    }
    // printf_("inside load image before resizing the image\n");
    if ((h&&w) && (H!=img.h || W!=img.w)) {
        // printf_("this image needs resizing\n");
        resize_image(&img, H, W);
    }
        
    
    if (is_h_flip) {
        // printf_("this image needs flipping\n");
        horizontal_flip(&img);
    }

    return img;
}


void get_next_batch(int n, float *X, int *Y, 
                    int w, int h, int c, int CLASSES )
{
    /**
     * sample next batch of data for training model
     * * Input:
     * n
     * w, h, c
     * CLASSES
     * Output:
     * X   [n,c,h,w]
     * Y   [n]
     * */
    (void)CLASSES;
    if (n <= 0) return;

    int imagesize_floats = w * h * c;
    int total = FC_DATASET_COUNT;
    if (total <= 0) return;

    for (int i = 0; i < n; i++)
    {
        int idx = (dataset_cursor + i) % total;
        int x_off = i * imagesize_floats;
        int copy_units = (imagesize_floats < FC_INPUT_UNITS) ? imagesize_floats : FC_INPUT_UNITS;

        memset(X + x_off, 0, (size_t)imagesize_floats * sizeof(float));
        memcpy(X + x_off, test_inputs + idx * FC_INPUT_UNITS, (size_t)copy_units * sizeof(float));
        Y[i] = test_labels[idx];
    }
    dataset_cursor = (dataset_cursor + n) % total;
}



void get_same_batch(int n, float *X, int *Y, 
                        int w, int h, int c, int CLASSES )
{
    /**
     * sample next batch of data for training model
     * 
     * Input:
     *      n
     *      w, h, c
     *      CLASSES
     *      fp
     * Output:
     *      X   [n,c,h,w]
     *      Y   [n]
     * */
    (void)CLASSES;
    if (n <= 0) return;

    int total = FC_DATASET_COUNT;
    if (total <= 0) return;

    int imagesize_floats = w * h * c;

    // Pin the batch window once so every call reuses the same samples.
    static int fixed_batch_start = -1;
    if (fixed_batch_start < 0 || fixed_batch_start >= total)
        fixed_batch_start = dataset_cursor % total;

    for (int i = 0; i < n; i++)
    {
        int idx = (fixed_batch_start + i) % total;
        int x_off = i * imagesize_floats;
        int copy_units = (imagesize_floats < FC_INPUT_UNITS) ? imagesize_floats : FC_INPUT_UNITS;

        memset(X + x_off, 0, (size_t)imagesize_floats * sizeof(float));
        memcpy(X + x_off, test_inputs + idx * FC_INPUT_UNITS, (size_t)copy_units * sizeof(float));
        Y[i] = test_labels[idx];
    }

}








void get_random_batch(int n, float *X, int *Y, 
                        int w, int h, int c, int CLASSES)
{
    //
    // not recommended
    //
    /**
     * sample random batch of data
     * 
     * Input:
     *      n
     *      w, h, c
     *      CLASSES
     * Output:
     *      X   [n,c,h,w]
     *      Y   [n]
     * */
    int imagesize_floats = w * h * c;
    int total = FC_DATASET_COUNT;
    if (total <= 0) return;

    for (int i = 0; i < n; i++)
    {
        int idx = rand() % total;
        int x_off = i * imagesize_floats;
        int copy_units = (imagesize_floats < FC_INPUT_UNITS) ? imagesize_floats : FC_INPUT_UNITS;

        memset(X + x_off, 0, (size_t)imagesize_floats * sizeof(float));
        memcpy(X + x_off, test_inputs + idx * FC_INPUT_UNITS, (size_t)copy_units * sizeof(float));
        Y[i] = test_labels[idx];
    }
}


int get_dataset_count(void)
{
    return FC_DATASET_COUNT;
}
