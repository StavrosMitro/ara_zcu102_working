// fconv_cmp: correctness + cycle comparison harness for fconv3d_CHx3x3_f16
// dataflow variants.
//
// Primary comparison is against fconv3d_CHx3x3_f16 (the existing block-per-
// row kernel), not a fresh scalar derivation of 3x3xC convolution -- that
// math isn't in question, only whether a new dataflow variant computes the
// identical function on the identical padded buffers. See kernel/fconv3d.h
// (symlinked from conv_layer16only/kernel/, the single source of truth --
// new variants belong there, never copied into this app).
//
// scalar_conv_ref below is a second, independent oracle kept deliberately
// cheap and structurally unrelated to either vector kernel. It earned its
// keep immediately: the first run of this harness showed fconv3d_CHx3x3_f16_
// stream disagreeing with fconv3d_CHx3x3_f16 at the last 2 columns of every
// row, and scalar_conv_ref is what proved the *block* kernel was the wrong
// one (a stale vfslide1down chain in its row1 tap, fixed in
// fconv3c_3x3x3_f32.c) -- "already trusted" was not actually true. Kept
// permanently rather than trusting either vector kernel unconditionally.

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

#define MAX_C 2
#define MAX_M 20
#define MAX_N 29

static _Float16 in_buf[MAX_C * (MAX_M + 2) * (MAX_N + 2)];
static _Float16 filt_buf[MAX_C * 9];
static _Float16 gold_buf[MAX_M * MAX_N];
static _Float16 var_buf[MAX_M * MAX_N];
static _Float16 ref_buf[MAX_M * MAX_N]; // debug-only scalar reference

static void scalar_conv_ref(int64_t M, int64_t N, int64_t C, _Float16 bias)
{
    const int64_t W = N + 2;
    for (int64_t r = 0; r < M; ++r) {
        for (int64_t c = 0; c < N; ++c) {
            float acc = (float)bias;
            for (int64_t ch = 0; ch < C; ++ch) {
                const _Float16 *chan = in_buf + ch * (M + 2) * W;
                const _Float16 *fc = filt_buf + ch * 9;
                for (int64_t kr = 0; kr < 3; ++kr) {
                    for (int64_t kc = 0; kc < 3; ++kc) {
                        acc += (float)chan[(r + kr) * W + (c + kc)] * (float)fc[kr * 3 + kc];
                    }
                }
            }
            ref_buf[r * N + c] = (_Float16)acc;
        }
    }
}

static inline int64_t cycle_count_local(void)
{
    int64_t c = 0;
    asm volatile("fence; csrr %0, cycle" : "=r"(c));
    return c;
}

// Zero-pads the border and fills the interior with a small deterministic
// pattern -- no rand(): bare-metal SPIKE/FPGA builds here have no libc rand,
// and determinism makes a failing diff reproducible.
static void fill_padded_input(int64_t M, int64_t N, int64_t C)
{
    const int64_t W = N + 2;
    const int64_t H = M + 2;
    for (int64_t ch = 0; ch < C; ++ch) {
        for (int64_t r = 0; r < H; ++r) {
            for (int64_t c = 0; c < W; ++c) {
                int64_t idx = ch * H * W + r * W + c;
                if (r == 0 || r == H - 1 || c == 0 || c == W - 1) {
                    in_buf[idx] = (_Float16)0.0f;
                } else {
                    int v = (int)((r * 13 + c * 7 + ch * 5) % 9) - 4;
                    in_buf[idx] = (_Float16)(v * 0.125f);
                }
            }
        }
    }
}

static void fill_filter(int64_t C)
{
    for (int64_t ch = 0; ch < C; ++ch) {
        for (int64_t k = 0; k < 9; ++k) {
            int v = (int)((k * 3 + ch * 11) % 7) - 3;
            filt_buf[ch * 9 + k] = (_Float16)(v * 0.1f);
        }
    }
}

// Bit-exact equality of the FP16 storage pattern -- not "close enough".
// Answers "are the losses identical" precisely instead of by eyeballing
// %.6f output, which hides anything below 1e-6-ish.
static int bits_equal(_Float16 a, _Float16 b)
{
    union { _Float16 f; uint16_t u; } ua, ub;
    ua.f = a;
    ub.f = b;
    return ua.u == ub.u;
}

// Returns 0 on match (within threshold), else the number of mismatching
// elements (prints the first few, mirroring convolution_layer.c's
// verify_img2col_implementations idiom). Also flags -- separately -- whether
// the "oracle" (gold) itself disagrees with the independent scalar
// reference, since a gold/stream match is only meaningful if gold is right.
static int compare_outputs(int64_t M, int64_t N, float threshold)
{
    int mismatches = 0;
    int gold_vs_ref_mismatches = 0;
    int exact_matches = 0;
    _Float16 max_diff = (_Float16)0.0f;
    for (int64_t idx = 0; idx < M * N; ++idx) {
        _Float16 diff = (_Float16)fabsf((float)gold_buf[idx] - (float)var_buf[idx]);
        if ((float)diff > threshold) {
            printf("  mismatch[%ld] (row=%ld col=%ld): gold=%.6f stream=%.6f ref=%.6f diff=%.6f\n",
                   (long)idx, (long)(idx / N), (long)(idx % N),
                   (float)gold_buf[idx], (float)var_buf[idx], (float)ref_buf[idx], (float)diff);
            mismatches++;
        }
        if (diff > max_diff) max_diff = diff;
        if (fabsf((float)gold_buf[idx] - (float)ref_buf[idx]) > threshold)
            gold_vs_ref_mismatches++;
        if (bits_equal(gold_buf[idx], var_buf[idx])) exact_matches++;
    }
    printf("  max_abs_diff=%.6f (threshold=%.6f)\n", (float)max_diff, threshold);
    printf("  bit-exact (identical FP16 storage, not just within threshold): %d/%ld\n",
           exact_matches, (long)(M * N));
    if (gold_vs_ref_mismatches > 0) {
        printf("  WARNING: gold (block kernel) disagrees with the scalar reference at"
               " %d/%ld positions -- the oracle itself may be wrong, a gold/stream"
               " match does not mean either is correct.\n",
               gold_vs_ref_mismatches, (long)(M * N));
    }
    return mismatches;
}

static void run_case(const char *name, int64_t M, int64_t N, int64_t C, _Float16 bias)
{
    printf("\n---- case %s: M=%ld N=%ld C=%ld ----\n", name, (long)M, (long)N, (long)C);

    fill_padded_input(M, N, C);
    fill_filter(C);

    int64_t t0 = cycle_count_local();
    fconv3d_CHx3x3_f16(gold_buf, in_buf, filt_buf, M, N, C, bias);
    int64_t cyc_gold = cycle_count_local() - t0;

    t0 = cycle_count_local();
    fconv3d_CHx3x3_f16_stream(var_buf, in_buf, filt_buf, M, N, C, bias);
    int64_t cyc_stream = cycle_count_local() - t0;

    scalar_conv_ref(M, N, C, bias);

    int mismatches = compare_outputs(M, N, 5e-2f);

    // (float)(uint32_t)n, not (float)n: fcvt.s.l (int64->float) silently
    // returns a constant on this FPGA bitstream (RVD=0, cvfpu INT64 gap --
    // see AGENT_NOTES_FP16_4LANE.md 46.8). Cycle counts here fit in 32 bits.
    printf("  block (oracle) : %ld cycles\n", (long)cyc_gold);
    printf("  stream (variant): %ld cycles  (%.2fx oracle)\n",
           (long)cyc_stream, (float)(uint32_t)cyc_stream / (float)(uint32_t)cyc_gold);
    printf("  result: %s (%d/%ld mismatches)\n",
           mismatches == 0 ? "PASS" : "FAIL", mismatches, (long)(M * N));
}

// Prints how wide each kernel's vector register GROUP actually is at
// runtime: fconv3d_CHx3x3_f16 (block) issues all its vector ops at LMUL=2,
// fconv3d_CHx3x3_f16_stream at LMUL=4 -- same VLEN, different group width,
// so this is the concrete "elements per instruction" each one gets. VLEN
// itself is derived back out (VLMAX(e16,mX) = VLEN*X/16) rather than read
// from a CSR, since there isn't a portable one for it.
static void report_vlen(void)
{
    unsigned long int vlmax_m2, vlmax_m4;
    asm volatile("vsetvli %0, zero, e16, m2, ta, ma" : "=r"(vlmax_m2));
    asm volatile("vsetvli %0, zero, e16, m4, ta, ma" : "=r"(vlmax_m4));

    printf("\n-- vector register usage (this build) --\n");
    printf("  block  (fconv3d_CHx3x3_f16)        : e16,m2 -> VLMAX=%lu elements/insn\n",
           vlmax_m2);
    printf("  stream (fconv3d_CHx3x3_f16_stream)  : e16,m4 -> VLMAX=%lu elements/insn\n",
           vlmax_m4);
    printf("  implied VLEN (from VLMAX(e16,m2)*8) : %lu bits\n", vlmax_m2 * 8);
    printf("  implied VLEN (from VLMAX(e16,m4)*4) : %lu bits (should match above)\n",
           vlmax_m4 * 4);
}

int main(void)
{
    printf("\n=====================================\n");
    printf("=  fconv3d_CHx3x3_f16: block vs stream  =\n");
    printf("=====================================\n");

    report_vlen();

    // Deliberately tiny and not VLMAX-aligned: exercises the avl<vlmax tail
    // chunk with almost nothing else going on.
    run_case("tiny", 1, 1, 1, (_Float16)0.1f);

    // Single row, single channel, single chunk (31 elems << VLMAX): isolates
    // whether the tail-of-row (last valid columns) handling is wrong even
    // with no multi-chunk or cross-channel interaction at all.
    run_case("single-row", 1, MAX_N, 1, (_Float16)0.1f);

    // W=31 is prime, so it can't evenly divide any power-of-2 VLMAX -- this
    // forces the column-rotation (col = (col+r) mod W) to do real work across
    // a chunk boundary that does NOT line up with a row boundary, and
    // total=M*W=620 spans multiple VLMAX(e16,m4) chunks on every VLEN this
    // repo currently builds for (256 @ VLEN=1024, 512 @ VLEN=2048).
    run_case("multi-chunk", MAX_M, MAX_N, MAX_C, (_Float16)0.25f);

    return 0;
}
