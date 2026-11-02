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


inline void make_image_32(image *img, int w, int h, int c);
inline void free_image_32(image *img);
image load_image_32(const unsigned char *img_bytes, int W, int H, int channels, int is_h_flip);
void resize_image_32(image *im, int w, int h);

void get_random_batch_32(int n, float *X, int *Y, 
                        int w, int h, int c, int CLASSES);
void get_next_batch_32(int n, float *X, int *Y, 
                        int w, int h, int c, int CLASSES );
void get_same_batch_32(int n, float *X, int *Y,
                        int w, int h, int c, int CLASSES );
int get_dataset_count_32(void);
