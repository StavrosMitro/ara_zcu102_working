// pluto_conv_stream: correctness harness for the Pluto "stream" 3x3
// convolution (NCHW, stride 1, pad 1), hand-vectorised from the tiled Pluto
// output -- kernel/conv_nchw_stream.c.
//
// The same layer is computed four ways and compared element by element:
//   ref     scalar_conv_ref() below: the definition of the convolution, on
//           the UNPADDED input with explicit border tests. It shares no
//           buffer layout and no loop structure with any vector kernel.
//   tiled   conv_nchw_stream() + compact_stream_output()
//   dyn     conv_nchw_stream_dyn() + compact_stream_output()
//   fconv3d fconv3d_CHx3x3_f32(), the row-block kernel conv_layer32 uses
//           (kernel/fconv3c_3x3x3_f32.c and kernel/fconv3d.h are symlinks into
//           conv_layer32/kernel/, the same arrangement as fconv_cmp).
//
// conv_nchw_stream() has its tile baked in (STREAM_TILE = 16 elements at
// e32/m1), so it needs VLEN >= 512; VLEN = 512 is where the tile is exactly
// VLMAX. conv_nchw_stream_dyn() strip-mines with the vl vsetvli grants and
// must pass at every VLEN. The vl this run actually gets is printed first.
//
//   make bin/pluto_conv_stream.spike
//   make spike-run-pluto_conv_stream \
//        RISCV_SIM_OPT="--isa=rv64gcv_zfh --varch=vlen:512,elen:64"

#include <math.h>
#include <stdint.h>

#include "kernel/fconv3d.h"

#ifdef SPIKE
#include "printf.h"
#elif defined ARA_LINUX
#include <stdio.h>
#else
#include "printf.h"
#endif

// Last: defines the single-letter shape macros B, C, K, P, Q, S.
#include "kernel/conv_nchw_stream.h"

#if B != 1
#error "conv_nchw_stream() is the Pluto output for B = 1"
#endif

#define PADDED_LEN ((P + 2) * WP) // fconv3d's channel stride (no +2 tail)
#define MAX_PRINT 8
#define PLUTO_CHECKSUM (-6895.0) // ~/pluto_work/conv/work_stream_tiles.*/ref.txt

static float X[C][P][Q];             // unpadded input: the source of truth
static float in_pad[C * PADDED_LEN]; // fconv3d's input layout
static float O_ref[K][P][Q];
static float O_fconv[K][P][Q];

static void scalar_conv_ref(void)
{
    for (int k = 0; k < K; ++k) {
        for (int oj = 0; oj < P; ++oj) {
            for (int oi = 0; oi < Q; ++oi) {
                float acc = bias[k];
                for (int c = 0; c < C; ++c) {
                    for (int r = 0; r < 3; ++r) {
                        for (int s = 0; s < 3; ++s) {
                            int y = oj + r - 1;
                            int x = oi + s - 1;
                            if (y >= 0 && y < P && x >= 0 && x < Q)
                                acc += X[c][y][x] * W[k][c][r][s];
                        }
                    }
                }
                O_ref[k][oj][oi] = acc;
            }
        }
    }
}

static inline int64_t cycle_count_local(void)
{
    int64_t c = 0;
    asm volatile("fence; csrr %0, cycle" : "=r"(c));
    return c;
}

// The Pluto self-check data (conv_nchw_stream.c: in_val, w_val, b_val). Small
// integers, so the layer's checksum must be the Pluto reference.
static void fill_pluto(void)
{
    for (int c = 0; c < C; ++c)
        for (int y = 0; y < P; ++y)
            for (int x = 0; x < Q; ++x)
                X[c][y][x] = (float)((c * 5 + y * 3 + x) % 7 - 3);
    for (int k = 0; k < K; ++k) {
        bias[k] = (float)(k % 3 - 1);
        for (int c = 0; c < C; ++c)
            for (int r = 0; r < 3; ++r)
                for (int s = 0; s < 3; ++s)
                    W[k][c][r][s] = (float)((k * 3 + c * 5 + r * 7 + s * 11) % 5 - 2);
    }
}

static uint32_t lcg_state;

// Integer in [-n, n].
static int lcg_int(int n)
{
    lcg_state = lcg_state * 1664525u + 1013904223u;
    return (int)((lcg_state >> 8) % (uint32_t)(2 * n + 1)) - n;
}

// Non-periodic data: an index that is off by a multiple of the Pluto
// pattern's period (7 along x) cannot hide here. Every value is a multiple of
// a power of two (1/8, 1/16, 1/4), so -- like the integer data -- every
// partial sum is exact in fp32 whatever the order of the additions.
static void fill_random(void)
{
    lcg_state = 12345u;
    for (int c = 0; c < C; ++c)
        for (int y = 0; y < P; ++y)
            for (int x = 0; x < Q; ++x)
                X[c][y][x] = (float)lcg_int(8) * 0.125f;
    for (int k = 0; k < K; ++k) {
        bias[k] = (float)lcg_int(4) * 0.25f;
        for (int c = 0; c < C; ++c)
            for (int r = 0; r < 3; ++r)
                for (int s = 0; s < 3; ++s)
                    W[k][c][r][s] = (float)lcg_int(8) * 0.0625f;
    }
}

// Zero-pads X into both vector kernels' input layouts.
static void pad_inputs(void)
{
    for (int c = 0; c < C; ++c) {
        for (int y = 0; y < P + 2; ++y) {
            for (int x = 0; x < WP; ++x) {
                float v = (y >= 1 && y <= P && x >= 1 && x <= Q) ? X[c][y - 1][x - 1] : 0.0f;
                I[0][c][y * WP + x] = v;
                in_pad[c * PADDED_LEN + y * WP + x] = v;
            }
        }
        // The 2 garbage positions of the last stream row read past the channel.
        I[0][c][PADDED_LEN] = 0.0f;
        I[0][c][PADDED_LEN + 1] = 0.0f;
    }
}

// An unrelated value everywhere, so a position a kernel never writes shows up.
static void poison_stream_outputs(void)
{
    for (int k = 0; k < K; ++k) {
        for (int t = 0; t < S; ++t)
            F[0][k][t] = 12345.0f;
        for (int oj = 0; oj < P; ++oj)
            for (int oi = 0; oi < Q; ++oi)
                O[0][k][oj][oi] = 12345.0f;
    }
}

static void poison_fconv_output(void)
{
    for (int k = 0; k < K; ++k)
        for (int oj = 0; oj < P; ++oj)
            for (int oi = 0; oi < Q; ++oi)
                O_fconv[k][oj][oi] = 12345.0f;
}

// Exact comparison, no threshold: both data sets keep every partial sum
// exactly representable (see fill_*), so neither the summation order nor FMA
// fusing can change a result -- any difference is a wrong term, not rounding.
// Returns the number of mismatching elements and prints the first few, with
// their position in the stream and in its tile.
static int compare(const char *what, const float *a, const float *b)
{
    int mismatches = 0;
    float max_diff = 0.0f;
    for (int idx = 0; idx < K * P * Q; ++idx) {
        if (a[idx] == b[idx])
            continue;
        float diff = fabsf(a[idx] - b[idx]);
        if (diff > max_diff) max_diff = diff;
        if (mismatches < MAX_PRINT) {
            int k = idx / (P * Q);
            int oj = (idx / Q) % P;
            int oi = idx % Q;
            int t = oj * WP + oi;
            printf("    k=%d oj=%d oi=%d (stream t=%d, tile lane %d): %.4f vs %.4f\n",
                   k, oj, oi, t, t % STREAM_TILE, a[idx], b[idx]);
        }
        mismatches++;
    }
    printf("  %s: %s (%d/%d mismatches, max_abs_diff=%.4f)\n", what,
           mismatches == 0 ? "PASS" : "FAIL", mismatches, K * P * Q, max_diff);
    return mismatches;
}

// The checksum of the Pluto self-check, taken on the compacted output.
static double pluto_checksum(void)
{
    double sum = 0.0;
    for (int k = 0; k < K; ++k)
        for (int y = 0; y < P; ++y)
            for (int x = 0; x < Q; ++x)
                sum += (double)O[0][k][y][x] * (double)((k * 31 + y * 7 + x) % 13 + 1);
    return sum;
}

// Runs one stream kernel + compact_stream_output() into F and O, and checks O
// against the reference and against fconv3d (both already computed). On Spike
// the cycle CSR counts retired instructions, not Ara cycles. Returns the
// number of failed checks.
static int check_stream(const char *label, void (*kernel)(void), int is_pluto_data)
{
    printf("  %s\n", label);
    poison_stream_outputs();

    int64_t t0 = cycle_count_local();
    kernel();
    int64_t cyc_kernel = cycle_count_local() - t0;

    t0 = cycle_count_local();
    compact_stream_output();
    int64_t cyc_compact = cycle_count_local() - t0;

    int fails = 0;
    fails += compare("  vs ref    ", &O[0][0][0][0], &O_ref[0][0][0]) != 0;
    fails += compare("  vs fconv3d", &O[0][0][0][0], &O_fconv[0][0][0]) != 0;

    if (is_pluto_data) {
        double sum = pluto_checksum();
        printf("    checksum %.1f (Pluto reference %.1f): %s\n", sum, PLUTO_CHECKSUM,
               sum == PLUTO_CHECKSUM ? "PASS" : "FAIL");
        fails += sum != PLUTO_CHECKSUM;
    }

    printf("    cycles: kernel %ld + compact %ld\n", (long)cyc_kernel, (long)cyc_compact);
    return fails;
}

// Returns the number of failed checks.
static int run_case(const char *name, void (*fill)(void), int is_pluto_data)
{
    printf("\n---- case %s: C=%d K=%d P=%d Q=%d ----\n", name, C, K, P, Q);

    fill();
    pad_inputs();
    scalar_conv_ref();

    printf("  fconv3d (conv_layer32)\n");
    poison_fconv_output();
    int64_t t0 = cycle_count_local();
    for (int k = 0; k < K; ++k)
        fconv3d_CHx3x3_f32(&O_fconv[k][0][0], in_pad, &W[k][0][0][0], P, Q, C, bias[k]);
    int64_t cyc_fconv = cycle_count_local() - t0;

    int fails = 0;
    fails += compare("  vs ref    ", &O_fconv[0][0][0], &O_ref[0][0][0]) != 0;
    printf("    cycles: kernel %ld\n", (long)cyc_fconv);

    fails += check_stream("stream, tile 16 (conv_nchw_stream) + compact",
                          conv_nchw_stream, is_pluto_data);
    fails += check_stream("stream, dynamic strips (conv_nchw_stream_dyn) + compact",
                          conv_nchw_stream_dyn, is_pluto_data);
    return fails;
}

// Prints the vl each stream kernel gets from its own vsetvli on this run.
static void report_vl(void)
{
    unsigned long int vlmax, vl;
    asm volatile("vsetvli %0, zero, e32, m1, ta, ma" : "=r"(vlmax));
    asm volatile("vsetvli %0, %1, e32, m1, ta, ma" : "=r"(vl) : "r"((unsigned long int)STREAM_TILE));

    printf("\n-- vector length (this run) --\n");
    printf("  VLMAX(e32,m1) = %lu elements -> VLEN = %lu bits\n", vlmax, vlmax * 32);
    printf("  conv_nchw_stream: vsetvli(avl=%d, e32, m1) -> vl = %lu\n", STREAM_TILE, vl);
    if (vl != (unsigned long int)STREAM_TILE)
        printf("  *** vl != tile: lanes %lu..%d of every tile are never computed,"
               " expect FAIL ***\n", vl, STREAM_TILE - 1);
    else if (vlmax == (unsigned long int)STREAM_TILE)
        printf("  tile == VLMAX: the right vl, every vector register is full\n");
    else
        printf("  tile < VLMAX: correct, but only %d of %lu lanes are used\n",
               STREAM_TILE, vlmax);
    printf("  conv_nchw_stream_dyn: %lu strips of %lu", (unsigned long int)S / vlmax, vlmax);
    if ((unsigned long int)S % vlmax != 0)
        printf(", then a short last strip of %lu", (unsigned long int)S % vlmax);
    printf(" (stream of %d)\n", S);
}

int main(void)
{
    printf("\n==========================================\n");
    printf("=  conv_nchw_stream vs reference/fconv3d  =\n");
    printf("==========================================\n");

    report_vl();

    int fails = 0;
    fails += run_case("pluto (integer pattern)", fill_pluto, 1);
    fails += run_case("random (dyadic fractions)", fill_random, 0);

    printf("\n==== %s ====\n", fails == 0 ? "ALL PASS" : "FAIL");
    return fails != 0;
}
