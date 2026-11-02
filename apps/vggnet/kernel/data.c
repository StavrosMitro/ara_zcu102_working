//
// File:        data.c
// Description: Provide functions for data process 
// Author:      Haris Wang
//
// #include <stdlib.h>
#include <stdio.h>
// #include <assert.h>
#include <string.h>
#include "data.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
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

static float image_scratch[IMAGE_SCRATCH_SLOTS][IMAGE_MAX_FLOATS];
static int image_scratch_cursor = 0;

static int dataset_cursor = 0;

#define MAX_DATASET_COUNT 60000
static int shuffle_indices[MAX_DATASET_COUNT];
static int shuffle_initialized = 0;

void dataset_shuffle(void)
{
    int total = cifar10_count;
    if (total <= 0) return;

    if (!shuffle_initialized) {
        for (int i = 0; i < total; i++) shuffle_indices[i] = i;
        shuffle_initialized = 1;
    }

    for (int i = total - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int tmp = shuffle_indices[i];
        shuffle_indices[i] = shuffle_indices[j];
        shuffle_indices[j] = tmp;
    }
    dataset_cursor = 0;
}

// ---------------------------------------------------------------------------
// Train/eval holdout split
//
// The dataset ships as one flat array. dataset_split_init() shuffles it once
// and permanently reserves the tail as the eval pool; the train pool is the
// head. Reshuffling between epochs only permutes the train pool, so no eval
// sample is ever trained on.
// ---------------------------------------------------------------------------

static int split_train_count = 0;
static int split_eval_count  = 0;
static int train_cursor = 0;
static int eval_cursor  = 0;

// augment: 0 = clean (eval), 1 = training augmentation. The horizontal flip
// reuses load_image's existing is_h_flip path (-> horizontal_flip).
static void load_sample(int idx, float *dst, int w, int h, int c, int augment)
{
    int n_floats = w * h * c;
    int flip = augment ? (rand() & 1) : 0;
    image img = load_image(cifar10_data + (unsigned int)idx * n_floats, w, h, c, flip);
    memcpy(dst, img.data, (size_t)n_floats * sizeof(float));
    free_image(&img);
}

void dataset_split_init(int eval_count)
{
    int total = cifar10_count;
    if (total <= 0) return;

    if (eval_count < 0)      eval_count = 0;
    if (eval_count >= total) eval_count = total / 5;

    dataset_shuffle();   // one global shuffle decides which samples are held out

    split_eval_count  = eval_count;
    split_train_count = total - eval_count;
    train_cursor = 0;
    eval_cursor  = 0;
}

int get_train_count(void) { return split_train_count; }
int get_eval_count(void)  { return split_eval_count;  }

void dataset_shuffle_train(void)
{
    for (int i = split_train_count - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int tmp = shuffle_indices[i];
        shuffle_indices[i] = shuffle_indices[j];
        shuffle_indices[j] = tmp;
    }
    train_cursor = 0;
}

static int g_augment = 0;
void dataset_set_augment(int on) { g_augment = on; }

void get_train_batch(int n, float *X, int *Y, int w, int h, int c)
{
    if (n <= 0 || split_train_count <= 0) return;

    int n_floats = w * h * c;
    for (int i = 0; i < n; i++) {
        int idx = shuffle_indices[(train_cursor + i) % split_train_count];
        Y[i] = cifar10_labels[idx];
        load_sample(idx, X + i * n_floats, w, h, c, g_augment);
    }
    train_cursor = (train_cursor + n) % split_train_count;
}

void eval_reset(void) { eval_cursor = 0; }

int get_eval_batch(int n, float *X, int *Y, int w, int h, int c)
{
    if (n <= 0 || split_eval_count <= 0) return 0;

    int valid = split_eval_count - eval_cursor;
    if (valid <= 0) return 0;
    if (valid > n) valid = n;

    int n_floats = w * h * c;
    for (int i = 0; i < valid; i++) {
        int idx = shuffle_indices[split_train_count + eval_cursor + i];
        Y[i] = cifar10_labels[idx];
        load_sample(idx, X + i * n_floats, w, h, c, 0);   // never augment eval
    }

    // Pad the tail of a partial batch so the forward pass never reads stale
    // memory. The caller scores only the first `valid` predictions.
    for (int i = valid; i < n; i++) {
        Y[i] = Y[valid - 1];
        memcpy(X + i * n_floats, X + (valid - 1) * n_floats,
               (size_t)n_floats * sizeof(float));
    }

    eval_cursor += valid;
    return valid;
}


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

    image img;
    int imagesize_floats = w * h * c; // Size for the output float array
    int image_bytes = w * h * c;      // Size of the raw image in the binary array (1 byte per pixel channel)

    int total = cifar10_count;
    if (total <= 0) return;

    if (!shuffle_initialized) {
        for (int i = 0; i < total; i++) shuffle_indices[i] = i;
        shuffle_initialized = 1;
    }

    for (int i = 0; i < n; i++)
    {
        int pos = (dataset_cursor + i) % total;
        int idx = shuffle_indices[pos];
        unsigned int start = idx * image_bytes;

        Y[i] = cifar10_labels[idx];

        img = load_image(cifar10_data + start, w, h, c, 0);
        memcpy(X + i * imagesize_floats, img.data, imagesize_floats * sizeof(float));

        free_image(&img);
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

    int total = cifar10_count;
    if (total <= 0) return;

    int imagesize_floats = w * h * c;
    int image_bytes = w * h * c;

    // Pin the batch window once so every call reuses the same samples.
    static int fixed_batch_start = -1;
    if (fixed_batch_start < 0 || fixed_batch_start >= total)
        fixed_batch_start = dataset_cursor % total;

    for (int i = 0; i < n; i++)
    {
        int idx = (fixed_batch_start + i) % total;
        unsigned int start = idx * image_bytes;

        Y[i] = cifar10_labels[idx];

        image img = load_image(cifar10_data + start, w, h, c, 0);
        memcpy(X + i * imagesize_floats, img.data, imagesize_floats * sizeof(float));
        free_image(&img);
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
    image img;
    make_image(&img, w, h, c);
    for (int i = 0; i < n; i++)
    {
        int idx = rand() % cifar10_count;
        unsigned int start = cifar10_offsets[idx];
        img = load_image(cifar10_data + start, w, h, c, 0);
        memcpy(X+i*w*h*c, img.data, w*h*c*sizeof(float));
        Y[i] = cifar10_labels[idx];
    }
    free_image(&img);
}


int get_dataset_count(void)
{
    return cifar10_count;
}
