//
// File:        data.h
// Description: interface of data process functions
// Author:      Haris Wang
//
#include <stdlib.h>

typedef struct{
    int w;
    int h;
    int c;
    float *data;
} image;


inline void make_image(image *img, int w, int h, int c);
inline void free_image(image *img);
image load_image(const unsigned char *img_bytes, int W, int H, int channels, int is_h_flip);
void resize_image(image *im, int w, int h);

void get_random_batch(int n, float *X, int *Y, 
                        int w, int h, int c, int CLASSES);
void get_next_batch(int n, float *X, int *Y, 
                        int w, int h, int c, int CLASSES );
void get_same_batch(int n, float *X, int *Y,
                        int w, int h, int c, int CLASSES );
int get_dataset_count(void);
void dataset_shuffle(void);

// Train/eval holdout split. dataset_split_init() reserves eval_count samples
// that get_train_batch() will never return.
void dataset_split_init(int eval_count);
int  get_train_count(void);
int  get_eval_count(void);
void dataset_shuffle_train(void);
void get_train_batch(int n, float *X, int *Y, int w, int h, int c);
void eval_reset(void);
int  get_eval_batch(int n, float *X, int *Y, int w, int h, int c);

// Training-time data augmentation: random horizontal flip + random crop with
// zero-padding (pad 4). Applied only by get_train_batch; eval is never
// augmented. Off by default.
void dataset_set_augment(int on);
