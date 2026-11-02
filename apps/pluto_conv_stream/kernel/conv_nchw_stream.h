// Shape and buffers of the Pluto "stream" 3x3 convolution (NCHW, stride 1,
// pad 1), named as in ~/pluto_work/conv/conv_nchw_stream.c.
//
// B, C, K, P, Q and S are single-letter macros: include this header AFTER any
// header that uses those letters as identifiers (kernel/fconv3d.h does).

#ifndef CONV_NCHW_STREAM_H
#define CONV_NCHW_STREAM_H

#define B 1
#define C 32
#define K 32
#define P 32
#define Q 32

#define WP (Q + 2) // padded row width
#define S (P * WP) // stream positions per output channel

// Tile of the vector loop baked into conv_nchw_stream() (Pluto tile.sizes).
// The kernel issues one "vsetvli avl=STREAM_TILE, e32, m1" per tile and never
// looks at the vl it gets back, so it is only correct when that vl is
// STREAM_TILE: VLMAX(e32,m1) >= 16, i.e. VLEN >= 512.
#define STREAM_TILE 16

extern float I[B][C][(P + 2) * WP + 2];
extern float W[K][C][3][3];
extern float bias[K];
extern float F[B][K][S];    // stream output, WP positions per row
extern float O[B][K][P][Q]; // compacted output

void conv_nchw_stream(void);     // fixed tile of STREAM_TILE
void conv_nchw_stream_dyn(void); // dynamic strip-mining: correct at any VLEN
void compact_stream_output(void);

#endif
