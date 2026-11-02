// Pluto "stream" 3x3 convolution, tiled for a 16-element vector (tile.sizes
// "1 16 32 65 193") and hand-vectorised: e32, m1, one tile per vsetvli.
//
// conv_nchw_stream() is the function from the Pluto variant
// fp16_v256_m1_uk1/out_vectorized.c, unchanged. It writes F at the padded row
// width (WP positions per output row, the last 2 of them garbage);
// compact_stream_output() keeps the Q real ones.

#include <math.h>
#include <stddef.h>

#include "conv_nchw_stream.h"

// Pluto declares every tN whether or not the schedule uses it, and its
// vectorisation pragmas are icc's.
// #pragma GCC diagnostic ignored "-Wunused-variable"
// #pragma GCC diagnostic ignored "-Wunused-but-set-variable"
// #pragma GCC diagnostic ignored "-Wunknown-pragmas"

#define ceild(n,d)  (((n)<0) ? -((-(n))/(d)) : ((n)+(d)-1)/(d))
#define floord(n,d) (((n)<0) ? -((-(n)+(d)-1)/(d)) : (n)/(d))
#define max(x,y)    ((x) > (y)? (x) : (y))
#define min(x,y)    ((x) < (y)? (x) : (y))

float I[1][32][(32 + 2) * (32 + 2) + 2];
float W[32][32][3][3];
float bias[32];
float F[1][32][(32 * (32 + 2))];
float O[B][K][P][Q];

void conv_nchw_stream(void)
{
  int b, k, t, c, r, s;
  (void)b;
  int t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11;
 register int lbv, ubv;
for (t2=0;t2<=31;t2++) {
  for (t3=0;t3<=67;t3++) {
    lbv=16*t3;
    ubv=16*t3+15;
// #pragma ivdep
// #pragma vector always
    for (t5=lbv;t5<=ubv;t5++) {
      F[0][t2][t5] = bias[t2];;
    }
  }
}
for (t2=0;t2<=31;t2++) {
  for (t3=0;t3<=67;t3++) {
    lbv=16*t3;
    ubv=16*t3+15;
    size_t avl = (size_t)(ubv - lbv + 1);
    size_t vl;

    asm volatile(
        "vsetvli %0, %1, e32, m1, ta, ma"
        : "=r"(vl)
        : "r"(avl)
    );
    asm volatile("vle32.v v0, (%0)" :: "r"(&F[0][t2][lbv]));

    for (t8=0;t8<=31;t8++) {
      for (t9=2*t8;t9<=2*t8+2;t9++) {
        
        // v0 = F[0][t2][lbv..ubv]: the accumulator, paired with the vse32 below
        for (t10=2*t8+2*t9;t10<=2*t8+2*t9+2;t10++) {
          asm volatile("vle32.v v1, (%0)" :: "r"(&I[0][t8][lbv + 34 * (-2*t8+t9) + (-2*t8-2*t9+t10)]));
// #pragma ivdep
// #pragma vector always
          asm volatile("vfmacc.vf v0, %0, v1" :: "f"(W[t2][t8][(-2*t8+t9)][(-2*t8-2*t9+t10)]));
          // for (t11=lbv;t11<=ubv;t11++) {
          //   F[0][t2][t11] += (I[0][t8][t11 + 34 * (-2*t8+t9) + (-2*t8-2*t9+t10)] * W[t2][t8][(-2*t8+t9)][(-2*t8-2*t9+t10)]);;
          // }
        }
        
      }

    }
    asm volatile("vse32.v v0, (%0)" :: "r"(&F[0][t2][lbv]));
  }
}
}

// conv_nchw_stream() with dynamic strip-mining instead of the fixed tile: the
// t3 tile loop and its lbv/ubv become one strip loop that asks vsetvli for
// whatever is left of the stream and advances by the vl it grants. The strip
// is VLMAX at any VLEN, and the last one is simply shorter. The c/r/s loops
// and the vector statements are the tiled version's, unchanged.
void conv_nchw_stream_dyn(void)
{S
  int t2, t5, t8, t9, t10;
  int lbv;
  size_t avl, vl;
  for (t2=0;t2<=31;t2++) {
    for (t5=0;t5<S;t5++) {
      F[0][t2][t5] = bias[t2];
    }
  }
  for (t2=0;t2<=31;t2++) {
    for (lbv=0;lbv<S;lbv+=(int)vl) {
      avl = (size_t)(S - lbv);

      // One vsetvli per strip: vl and vtype stay valid for all the taps below.
      asm volatile(
          "vsetvli %0, %1, e32, m1, ta, ma"
          : "=r"(vl)
          : "r"(avl)
      );
              // v0 = F[0][t2][lbv..lbv+vl-1]: the accumulator, paired with the vse32 below
      asm volatile("vle32.v v0, (%0)" :: "r"(&F[0][t2][lbv]));
      for (t8=0;t8<=31;t8++) {
        for (t9=2*t8;t9<=2*t8+2;t9++) {

          for (t10=2*t8+2*t9;t10<=2*t8+2*t9+2;t10++) {
            asm volatile("vle32.v v1, (%0)" :: "r"(&I[0][t8][lbv + 34 * (-2*t8+t9) + (-2*t8-2*t9+t10)]));
            asm volatile("vfmacc.vf v0, %0, v1" :: "f"(W[t2][t8][(-2*t8+t9)][(-2*t8-2*t9+t10)]));
          }
        }
      }
      asm volatile("vse32.v v0, (%0)" :: "r"(&F[0][t2][lbv]));
    }
  }
}

void conv_nchw_stream_dyn_m8(void)
{S
  int t2, t5, t8, t9, t10;
  int lbv;
  size_t avl, vl;
  for (t2=0;t2<=31;t2++) {
    for (t5=0;t5<S;t5++) {
      F[0][t2][t5] = bias[t2];
    }
  }
  for (t2=0;t2<=31;t2++) {
    for (lbv=0;lbv<S;lbv+=(int)vl) {
      avl = (size_t)(S - lbv);

      // One vsetvli per strip: vl and vtype stay valid for all the taps below.
      asm volatile(
          "vsetvli %0, %1, e32, m8, ta, ma"
          : "=r"(vl)
          : "r"(avl)
      );
              // v0 = F[0][t2][lbv..lbv+vl-1]: the accumulator, paired with the vse32 below
      asm volatile("vle32.v v0, (%0)" :: "r"(&F[0][t2][lbv]));
      for (t8=0;t8<=31;t8++) {
        for (t9=2*t8;t9<=2*t8+2;t9++) {

          for (t10=2*t8+2*t9;t10<=2*t8+2*t9+2;t10++) {
            asm volatile("vle32.v v8, (%0)" :: "r"(&I[0][t8][lbv + 34 * (-2*t8+t9) + (-2*t8-2*t9+t10)]));
            asm volatile("vfmacc.vf v0, %0, v8" :: "f"(W[t2][t8][(-2*t8+t9)][(-2*t8-2*t9+t10)]));
          }
        }
      }
      asm volatile("vse32.v v0, (%0)" :: "r"(&F[0][t2][lbv]));
    }
  }
}

void compact_stream_output(void)
{
    for (int b = 0; b < B; ++b)
        for (int k = 0; k < K; ++k)
            for (int oj = 0; oj < P; ++oj)
                for (int oi = 0; oi < Q; ++oi)
                    O[b][k][oj][oi] = F[b][k][oj * WP + oi];
}
