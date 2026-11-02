// vec_test - minimal Ara FP16 vector instruction check.
//
// vggnet16's conv1 produces 393216/393216 NaN (0x7e00) on BOTH 2 and 4 lanes,
// while a scalar FP16 reference on the same core is finite and the true
// magnitude is 2 (FP16 overflows above 65504). This isolates the exact
// instructions the conv kernel uses, on known constants, with no network,
// no dataset and no scratch buffers.
//
// Build: -DFPGA=1 -DNO_BOOT_LLC_FLUSH=1  (the second is mandatory or it hangs
// before main). Output appears only when the app exits (crt0 _eoc LLC flush).

#include <stdint.h>
#include "printf.h"
#include "probe_rodata.h"

#define N 64
static _Float16 src[N], dst[N];

// bn1 reduces spatial_size = 1024 elements per channel at e16/m8. For
// VLEN=2048 that is exactly VLMAX (2048*8/16 = 1024), so bn1 performs a SINGLE
// full-register-group reduction at maximum vl -- while T8c only reached vl=64.
// This buffer lets T10/T11 sweep vl and LMUL up to bn1's actual shape.
#define NBIG 1024
static _Float16 big[NBIG];

// 1 MiB -- exactly the LLC data-array size, so a sweep over it touches every
// set and way. Static (not an absolute address) so Spike can run it too.
#define SWEEP_HW ((1024UL * 1024UL) / 2UL)
static unsigned short sweep_buf[SWEEP_HW];

static inline unsigned bits16(_Float16 v) {
    union { _Float16 f; unsigned short u; } c; c.f = v; return (unsigned)c.u;
}
static int fails = 0;
static void chk(const char *nm, unsigned got, unsigned exp) {
    int ok = (got == exp);
    if (!ok) fails++;
    printf_("  [%s] %-26s got=0x%04x exp=0x%04x\n", ok ? "PASS" : "FAIL", nm, got, exp);
}
static void dump(const char *nm, int n) {
    printf_("    %s:", nm);
    for (int i = 0; i < n; i++) printf_(" %04x", bits16(dst[i]));
    printf_("\n");
}
// Emit the scalar divide/sqrt instructions directly rather than calling libm.
// CVA6's FPU has DIVSQRT enabled (fpu_wrap.sv: MERGED) while Ara's vector FPU
// has it DISABLED (vmfpu.sv), so it matters that these are the CVA6 scalar ops
// and that no compiler lowering can substitute a software routine for them.
static inline float hw_sqrtf(float x) {
    float r; asm volatile("fsqrt.s %0, %1" : "=f"(r) : "f"(x)); return r;
}
static inline float hw_divf(float a, float b) {
    float r; asm volatile("fdiv.s %0, %1, %2" : "=f"(r) : "f"(a), "f"(b)); return r;
}
static inline unsigned bits32(float v) {
    union { float f; unsigned u; } c; c.f = v; return c.u;
}
// Exact unsigned-integer -> fp16 bit pattern, computed with INTEGER ops only.
// The earlier harness used `(_Float16)(float)vl`, which returned 0x7c00 (+Inf)
// for every value on hardware while Spike computed it correctly -- i.e. the
// expectation itself was being produced by a suspect instruction
// (fcvt.s.lu / fcvt.s.l, exercised by T15 below). An expected value must never
// depend on hardware under test.
// Valid for 0 < v <= 2048, which covers every vl this test uses; every such
// integer is exactly representable in fp16.
static unsigned u2f16_bits(unsigned long v) {
    if (v == 0) return 0x0000;
    int e = 0;                      // find the MSB position
    unsigned long t = v;
    while (t > 1) { t >>= 1; e++; }
    // value = 1.mant * 2^e ; fp16 bias is 15, mantissa is 10 bits
    unsigned long mant = (v << (10 - e)) & 0x3FFUL;   // e <= 10 for v <= 2048
    return (unsigned)(((15 + e) << 10) | mant);
}

int main(void) {
    unsigned long vl;
    // FP16 constants: 1.0=0x3C00 2.0=0x4000 3.0=0x4200 7.0=0x4700 9.0=0x4880
    _Float16 one = 1.0f, two = 2.0f, three = 3.0f, nine = 9.0f;

    printf_("\n===== vec_test: Ara FP16 vector instruction check =====\n");
    asm volatile("vsetvli %0, %1, e16, m1, ta, ma" : "=r"(vl) : "r"((unsigned long)N));
    printf_("vl=%lu (e16,m1)  NR_LANES/VLEN are build-time only; vl is what matters\n", vl);

    // ---- T0: MEMORY INTEGRITY SWEEP -- run this FIRST ---------------------
    // Results have been drifting between builds on one bitstream: T10 gave
    // three different answers across three binaries, each binary internally
    // deterministic (two bit-identical runs). Deterministic-per-layout but
    // varying-with-layout is the signature of an ADDRESS-dependent fault, not
    // an instruction bug -- no property of vfredusum.vs can give two answers
    // for the same opcode, and T2-vs-T17a already showed byte-identical
    // instruction sequences disagreeing inside ONE run.
    //
    // The uninitialised-memory explanation is ruled out by construction:
    //   - .bss IS zero-filled in the flat .bin (.l2 is PROGBITS and follows
    //     every .bss section, so objcopy must pad the NOBITS gap; the .bin
    //     size equals the full span, and the pad bytes are verified zero)
    //   - every stack slot this program reads is also written by it
    // So the operands are sound and something in the memory path is not.
    //
    // This sweeps 1 MiB -- exactly the LLC data-array size, so it touches every
    // set and way -- storing an address-derived pattern, then reads it back.
    // A plain data error shows as scattered mismatches; an INDEXING error
    // shows as readback values belonging to a different address, and the
    // delta between expected and observed names the aliasing stride.
    // Uses a static 1 MiB buffer rather than an absolute address so the same
    // code is valid under Spike (whose memory map does not cover the board's
    // DRAM window) and can therefore be validated before it is trusted.
    printf_("\n[T0] memory integrity sweep (1 MiB, address-tagged)\n");
    {
        volatile unsigned short *m = (volatile unsigned short *)sweep_buf;
        const unsigned long NHW = SWEEP_HW;
        unsigned long bad = 0, firstbad = ~0UL;
        unsigned short gotv = 0, expv = 0;

        // Tag each halfword with its own index. Never 0x7e00 for any index,
        // so a stale-NaN read is distinguishable from a wrong-address read.
        for (unsigned long i = 0; i < NHW; i++)
            m[i] = (unsigned short)(i & 0xFFFFUL);

        for (unsigned long i = 0; i < NHW; i++) {
            unsigned short e = (unsigned short)(i & 0xFFFFUL);
            unsigned short g = m[i];
            if (g != e) {
                if (bad == 0) { firstbad = i; gotv = g; expv = e; }
                bad++;
            }
        }
        printf_("    mismatches=%lu of %lu halfwords\n", bad, NHW);
        if (bad) {
            long delta = (long)gotv - (long)expv;
            printf_("    first bad: index=%lu addr=0x%08lx got=0x%04x exp=0x%04x delta=%ld\n",
                    firstbad, (unsigned long)(uintptr_t)&sweep_buf[firstbad],
                    (unsigned)gotv, (unsigned)expv, delta);
            printf_("    (delta = observed-expected tag; if constant across\n"
                    "     mismatches it is the aliasing stride in halfwords)\n");
        }
        chk("T0 memory sweep clean", (unsigned)(bad != 0), 0u);
    }

    // ---- T1: vfmv.v.f  -- splat a scalar into a vector -------------------
    // If the .vf scalar operand path (incl. NaN-boxing) is broken, this alone fails.
    for (int i = 0; i < N; i++) dst[i] = (_Float16)0.0f;
    asm volatile("vfmv.v.f v8, %0" :: "f"(one));
    asm volatile("vse16.v v8, (%0)" :: "r"(dst));
    printf_("\n[T1] vfmv.v.f  (splat 1.0)\n");
    chk("dst[0]", bits16(dst[0]), 0x3C00);
    chk("dst[vl-1]", bits16(dst[vl-1]), 0x3C00);
    dump("first8", 8);

    // ---- T2: vfmacc.vf -- vd += f * vs2 ----------------------------------
    // vd=1.0, vs2=2.0, f=3.0  ->  1 + 3*2 = 7.0 = 0x4700
    asm volatile("vfmv.v.f v8, %0" :: "f"(two));    // vs2 = 2.0
    asm volatile("vfmv.v.f v16, %0" :: "f"(one));   // vd  = 1.0
    asm volatile("vfmacc.vf v16, %0, v8" :: "f"(three));
    asm volatile("vse16.v v16, (%0)" :: "r"(dst));
    printf_("\n[T2] vfmacc.vf  (1.0 + 3.0*2.0)\n");
    chk("dst[0]", bits16(dst[0]), 0x4700);
    chk("dst[vl-1]", bits16(dst[vl-1]), 0x4700);
    dump("first8", 8);

    // ---- T2b: is the OPERAND bad, or the INSTRUCTION? ---------------------
    // T2 and T5 return 0x7e00 while T17a -- a byte-identical reproduction of
    // T2 later in the same program -- passes. Same instructions, same scalars,
    // same vl, one run, different answers. So the fault is positional, not
    // instruction-determined, and no amount of copying T2's shape will
    // reproduce it. Stop inferring the operand and read it: dump the raw
    // f-register bits immediately before the vfmacc, and the vector inputs
    // too. If the scalars read back correct here and the result is still NaN,
    // the operand delivery is fine and the fault is downstream; if they read
    // back wrong, the constants themselves were corrupted before use.
    printf_("\n[T2b] operand readback at the point of failure\n");
    {
        unsigned b_one, b_two, b_three;
        asm volatile("fmv.x.w %0, %1" : "=r"(b_one)   : "f"(one));
        asm volatile("fmv.x.w %0, %1" : "=r"(b_two)   : "f"(two));
        asm volatile("fmv.x.w %0, %1" : "=r"(b_three) : "f"(three));
        printf_("    f-regs: one=0x%08x two=0x%08x three=0x%08x\n",
                b_one, b_two, b_three);
        printf_("    (FLEN=32 NaN-boxed expect: 0xffff3c00 / 0xffff4000 / 0xffff4200)\n");
        chk("T2b one boxed",   b_one,   0xffff3c00u);
        chk("T2b two boxed",   b_two,   0xffff4000u);
        chk("T2b three boxed", b_three, 0xffff4200u);

        // Re-run T2's exact sequence, dumping each vector stage.
        asm volatile("vfmv.v.f v8, %0" :: "f"(two));
        asm volatile("vse16.v v8, (%0)" :: "r"(dst));
        chk("T2b v8 = splat(2.0)", bits16(dst[0]), 0x4000);
        asm volatile("vfmv.v.f v16, %0" :: "f"(one));
        asm volatile("vse16.v v16, (%0)" :: "r"(dst));
        chk("T2b v16 = splat(1.0)", bits16(dst[0]), 0x3c00);
        asm volatile("vfmv.v.f v8, %0" :: "f"(two));
        asm volatile("vfmv.v.f v16, %0" :: "f"(one));
        asm volatile("vfmacc.vf v16, %0, v8" :: "f"(three));
        asm volatile("vse16.v v16, (%0)" :: "r"(dst));
        chk("T2b vfmacc result", bits16(dst[0]), 0x4700);
        dump("first8", 8);
    }

    // ---- T3: vle16.v / vse16.v round trip ---------------------------------
    for (int i = 0; i < N; i++) { src[i] = (_Float16)(float)i; dst[i] = (_Float16)0.0f; }
    asm volatile("vle16.v v8, (%0)" :: "r"(src));
    asm volatile("vse16.v v8, (%0)" :: "r"(dst));
    printf_("\n[T3] vle16.v/vse16.v round trip (ramp 0,1,2,...)\n");
    chk("dst[0]==0.0", bits16(dst[0]), 0x0000);
    chk("dst[1]==1.0", bits16(dst[1]), 0x3C00);
    chk("dst[2]==2.0", bits16(dst[2]), 0x4000);
    dump("first8", 8);

    // ---- T4: vfslide1down.vf  -- THE conv row-slide primitive --------------
    // vd[i] = vs2[i+1] ; vd[vl-1] = f.   vs2 = ramp -> vd = 1,2,3,... ,9.0
    asm volatile("vle16.v v8, (%0)" :: "r"(src));
    asm volatile("vfslide1down.vf v16, v8, %0" :: "f"(nine));
    asm volatile("vse16.v v16, (%0)" :: "r"(dst));
    printf_("\n[T4] vfslide1down.vf  (ramp shifted, 9.0 slid into the tail)\n");
    chk("dst[0]==1.0", bits16(dst[0]), 0x3C00);
    chk("dst[1]==2.0", bits16(dst[1]), 0x4000);
    chk("dst[vl-1]==9.0", bits16(dst[vl-1]), 0x4880);
    dump("first8", 8);

    // ---- T5: the exact conv inner pattern ---------------------------------
    // slide, then multiply-accumulate by a scalar tap -- what fconv3d does.
    asm volatile("vle16.v v8, (%0)" :: "r"(src));
    asm volatile("vfmv.v.f v16, %0" :: "f"(one));       // acc = 1.0 (bias)
    asm volatile("vfslide1down.vf v24, v8, %0" :: "f"(nine));
    asm volatile("vfmacc.vf v16, %0, v24" :: "f"(two)); // acc += 2.0 * slid
    asm volatile("vse16.v v16, (%0)" :: "r"(dst));
    printf_("\n[T5] slide + vfmacc (conv inner pattern)\n");
    chk("dst[0]==1+2*1=3", bits16(dst[0]), 0x4200);
    chk("dst[1]==1+2*2=5", bits16(dst[1]), 0x4500);
    dump("first8", 8);

    // ---- T6: SCALAR fp16 on CVA6 (no vector unit involved) ----------------
    // conv1's vector output now matches Spike bit-for-bit, yet the A/B probe
    // reported max_finite=2147483647 (INT_MAX -- the fcvt.w.s saturation
    // result) for a quantity built only from finite fp16 values, whose largest
    // representable magnitude is 65504. Spike prints 4 for the same loop. That
    // pins the fault on the scalar path, which is also where the whole loss
    // computation lives ((float)pp[i] on every logit). These sub-tests split
    // the scalar path into its three pieces so the next run says which one.
    //
    // Everything below is deliberately volatile: -ffast-math plus -O3 would
    // otherwise constant-fold each expression at compile time and test the
    // host compiler instead of the CVA6 FPU.
    printf_("\n[T6] scalar fp16 on CVA6 (widen / compare / arith)\n");
    {
        volatile _Float16 h4   = (_Float16)4.0f;      // 0x4400
        volatile _Float16 h2   = (_Float16)2.0f;      // 0x4000
        volatile _Float16 hneg = (_Float16)-3.5f;     // 0xc300

        // T6a: fcvt.s.h then fcvt.w.s -- the exact "(int)(float)vmax" chain
        // that produced INT_MAX on hardware.
        volatile float f4 = (float)h4;
        volatile int   i4 = (int)f4;
        chk("T6a widen f16->f32 bits", *(volatile unsigned *)&f4, 0x40800000u);
        chk("T6a f16->f32->int",       (unsigned)i4,              4u);

        // T6b: scalar fp16 compare (flt.h) -- drives the `a > vmax` update.
        volatile _Float16 a = hneg < (_Float16)0 ? (_Float16)-hneg : hneg;
        chk("T6b fabs(-3.5) bits", bits16(a), 0x4300);
        chk("T6b (4.0 > 2.0)",     (unsigned)(h4 > h2), 1u);
        chk("T6b (2.0 > 4.0)",     (unsigned)(h2 > h4), 0u);

        // T6c: scalar fp16 arithmetic (fadd.h / fmul.h).
        volatile _Float16 s = (_Float16)(h4 + h2);
        volatile _Float16 m = (_Float16)(h4 * h2);
        chk("T6c 4.0+2.0=6.0 bits", bits16(s), 0x4600);
        chk("T6c 4.0*2.0=8.0 bits", bits16(m), 0x4800);

        // T6d: the f32->f16 narrowing direction (fcvt.h.s), used to write
        // every gradient back after the fp32 softmax island.
        volatile float  g  = 0.5f;
        volatile _Float16 n = (_Float16)g;
        chk("T6d narrow f32->f16 bits", bits16(n), 0x3800);
    }

    // ---- T7: the ops BatchNorm needs that T6 did not cover ---------------
    // bn1 is the first layer to go non-finite while its input is bit-exact.
    // Its training path needs two things T6 never touched:
    //   (a) scalar fsqrt.s / fdiv.s  -- inv_std = 1.0f/sqrtf(var+EPSILON)
    //   (b) an fp16 SUBNORMAL as a .vf scalar operand -- `factor` is
    //       1/(batchsize*spatial_size) = 1/32768 = 2^-15, and fp16's smallest
    //       NORMAL is 2^-14, so for bn1 this scalar is subnormal. It scales
    //       both the mean and the variance. An FPU that flushes subnormals to
    //       zero silently zeroes every channel statistic.
    printf_("\n[T7] fsqrt/fdiv + fp16 subnormal .vf scalar\n");
    {
        // T7a: the scalar CVA6 sqrt and divide, emitted as real instructions.
        volatile float v16f = 16.0f;
        float r = hw_sqrtf(v16f);
        chk("T7a fsqrt.s(16.0)", bits32(r), 0x40800000u);          // 4.0
        volatile float one = 1.0f;
        float d = hw_divf(one, r);
        chk("T7a fdiv.s(1,4)",   bits32(d), 0x3e800000u);          // 0.25

        // The exact bn1 expression with var == 0: 1/sqrt(0 + 1e-5) = 316.2278.
        volatile float eps = 1e-5f;
        float rt  = hw_sqrtf(eps);
        float inv = hw_divf(one, rt);
        // Expected values taken from the Spike reference run, not derived by
        // hand: 1e-5f is not exactly 1e-5, so the correctly-rounded results
        // are 0x3b4f3e37 (3.16228e-3) and 0x439e1d28 (316.2278).
        chk("T7a sqrt(1e-5)",      bits32(rt),  0x3b4f3e37u);
        chk("T7a 1/sqrt(1e-5)",    bits32(inv), 0x439e1d28u);      // 316.2278
        chk("T7a  ... -> f16",     bits16((_Float16)inv), 0x5cf1u);

        // T7b: is 2^-15 subnormal-representable and does it SURVIVE?
        volatile float  f32_factor = 1.0f / 32768.0f;      // 2^-15 exactly
        volatile _Float16 factor   = (_Float16)f32_factor;
        chk("T7b 2^-15 -> f16 bits", bits16(factor), 0x0200);   // subnormal

        // T7c: that subnormal used as a .vf scalar, the way batchnorm scales
        // avg[] and var[]. 32768.0 * 2^-15 = 1.0 exactly.
        {
            _Float16 big = (_Float16)32768.0f;              // 0x7800
            for (int i = 0; i < N; i++) src[i] = big;
            asm volatile("vsetvli t0, %0, e16, m8, ta, ma" :: "r"((long)N) : "t0");
            asm volatile("vle16.v v8, (%0)" :: "r"(src));
            asm volatile("vfmul.vf v8, v8, %0" :: "f"(factor));
            asm volatile("vse16.v v8, (%0)" :: "r"(dst));
            chk("T7c 32768*2^-15 =1.0", bits16(dst[0]), 0x3c00);
            chk("T7c  ... at vl-1",     bits16(dst[N - 1]), 0x3c00);
            dump("first8", 8);
        }

        // T7d: the same multiply done SCALAR, to tell a vector-only subnormal
        // fault apart from a machine-wide one.
        volatile _Float16 sbig = (_Float16)32768.0f;
        volatile _Float16 sres = (_Float16)(sbig * factor);
        chk("T7d scalar 32768*2^-15", bits16(sres), 0x3c00);
    }

    // ---- T8: the reduction seed path (why bn1 dies) ----------------------
    // bn1's [A] stage returns 0x7e00 (canonical fp16 NaN) from vfredusum.vs on
    // entirely finite input. The reduction seeds its accumulator with
    //     fmv.h.x fa3, zero ; vfmv.s.f v16, fa3
    // and ara_dispatcher substitutes 0x7e00 for any OPFVF scalar that is not
    // NaN-boxed to FLEN (=32 here). So if fmv.h.x fails to NaN-box, the seed
    // is silently replaced by NaN and the whole reduction returns NaN --
    // exactly the observed value. T8a tests the boxing directly by reading the
    // f-register back as raw bits; T8b/T8c walk the same path the kernel uses.
    printf_("\n[T8] reduction seed: NaN-boxing + vfmv.s.f + vfredusum.vs\n");
    {
        // T8a: does fmv.h.x NaN-box? For FLEN=32 an fp16 0.0 must read back as
        // 0xFFFF0000. Reading 0x00000000 means boxing is broken and every
        // OPFVF scalar built this way becomes NaN in the dispatcher.
        unsigned raw;
        asm volatile("fmv.h.x fa5, zero\n\t"
                     "fmv.x.w %0, fa5" : "=r"(raw) :: "fa5");
        chk("T8a fmv.h.x(0) boxed", raw, 0xffff0000u);

        // Same for 1.0 (0x3c00), which T1 exercised indirectly and which passed.
        unsigned raw1;
        asm volatile("li t1, 0x3c00\n\t"
                     "fmv.h.x fa5, t1\n\t"
                     "fmv.x.w %0, fa5" : "=r"(raw1) :: "fa5", "t1");
        chk("T8a fmv.h.x(1.0) boxed", raw1, 0xffff3c00u);

        // T8b: vfmv.s.f / vfmv.f.s round trip of a zero seed. If T8a fails,
        // this returns 0x7e00 and names the mechanism outright.
        _Float16 seed = (_Float16)0.0f, back;
        asm volatile("vsetvli t0, %0, e16, m8, ta, ma" :: "r"((long)N) : "t0");
        asm volatile("vfmv.s.f v16, %0" :: "f"(seed));
        asm volatile("vfmv.f.s %0, v16" : "=f"(back));
        chk("T8b vfmv.s.f(0)->f.s", bits16(back), 0x0000);

        // T8c: the actual reduction. 64 lanes of 1.0 with a 0.0 seed = 64.0.
        for (int i = 0; i < N; i++) src[i] = (_Float16)1.0f;
        asm volatile("vsetvli t0, %0, e16, m8, ta, ma" :: "r"((long)N) : "t0");
        asm volatile("vle16.v v8, (%0)" :: "r"(src));
        asm volatile("vfmv.s.f v16, %0" :: "f"(seed));
        asm volatile("vfredusum.vs v16, v8, v16");
        asm volatile("vfmv.f.s %0, v16" : "=f"(back));
        chk("T8c vfredusum 64x1.0", bits16(back), 0x5400);          // 64.0

        // T8d: same reduction, but seed loaded from memory with flh instead of
        // built with fmv.h.x. flh NaN-boxes by definition, so if T8c fails and
        // T8d passes, the fault is specifically fmv.h.x, not the reduction.
        static _Float16 zero_mem;                 /* = 0.0, in .bss */
        zero_mem = (_Float16)0.0f;
        _Float16 seed2;
        asm volatile("flh %0, 0(%1)" : "=f"(seed2) : "r"(&zero_mem));
        asm volatile("vsetvli t0, %0, e16, m8, ta, ma" :: "r"((long)N) : "t0");
        asm volatile("vle16.v v8, (%0)" :: "r"(src));
        asm volatile("vfmv.s.f v16, %0" :: "f"(seed2));
        asm volatile("vfredusum.vs v16, v8, v16");
        asm volatile("vfmv.f.s %0, v16" : "=f"(back));
        chk("T8d vfredusum flh seed", bits16(back), 0x5400);
    }

    // ---- T9: scalar fdiv.s, the second fault T7a exposed -----------------
    // T7a found fdiv.s(1.0, 4.0) correct but 1.0/sqrt(1e-5) (= 1/3.162e-3,
    // i.e. 316.23) returning 0x00000000. Sweep the divisor magnitude to map
    // where the divider breaks; bn1 never reaches a divide, so this is a
    // separate defect that will bite bn5 backward.
    printf_("\n[T9] fdiv.s magnitude sweep (1.0 / x)\n");
    {
        volatile float one = 1.0f;
        static const struct { float x; unsigned exp; const char *nm; } cases[] = {
            { 4.0f,        0x3e800000u, "1/4        =0.25"     },
            { 0.5f,        0x40000000u, "1/0.5      =2.0"      },
            { 0.0625f,     0x41800000u, "1/0.0625   =16.0"     },
            { 0.00390625f, 0x43800000u, "1/2^-8     =256.0"    },
            { 0.001f,      0x4479ffffu, "1/0.001    =999.99994" },
            { 3.1622777e-3f, 0x439e1d28u, "1/3.1623e-3=316.23" },
            { 1e-4f,       0x461c4000u, "1/1e-4     =10000.0"  },
        };
        for (unsigned i = 0; i < sizeof(cases)/sizeof(cases[0]); i++) {
            volatile float xv = cases[i].x;
            float q = hw_divf(one, xv);
            chk(cases[i].nm, bits32(q), cases[i].exp);
        }
    }

    // ---- T10/T11: vfredusum.vs at bn1's ACTUAL shape ---------------------
    // T8c reduced 64 elements and passed; bn1 reduces 1024 at e16/m8, which is
    // exactly VLMAX for VLEN=2048. The only structural difference between the
    // passing minimal case and the failing kernel is vl and the register-group
    // width, so sweep both and find the threshold where the result goes wrong.
    // A reduction of vl copies of 1.0 must equal (float)vl, and fp16 represents
    // every integer up to 2048 exactly, so all expectations below are exact.
    {
        _Float16 zero16 = (_Float16)0.0f;
        unsigned long rvl;
        _Float16 res;

        // vl sweep at LMUL=8, all elements 1.0.
        printf_("\n[T10] vfredusum.vs vl sweep @ e16,m8 (sum of vl x 1.0)\n");
        for (int i = 0; i < NBIG; i++) big[i] = (_Float16)1.0f;
#define RED_M8(req, nm)                                                        \
        do {                                                                   \
            asm volatile("vsetvli %0, %1, e16, m8, ta, ma"                      \
                         : "=r"(rvl) : "r"((long)(req)));                       \
            asm volatile("vle16.v v8, (%0)"  :: "r"(big));                      \
            asm volatile("vfmv.s.f v16, %0"  :: "f"(zero16));                   \
            asm volatile("vfredusum.vs v16, v8, v16");                          \
            asm volatile("vfmv.f.s %0, v16"  : "=f"(res));                      \
            printf_("    (requested %d, vl granted = %lu)\n", (int)(req), rvl); \
            chk(nm, bits16(res), u2f16_bits(rvl));                              \
        } while (0)
        // Expectations are derived from the vl the hardware actually GRANTS,
        // not from the requested count: VLMAX depends on VLEN, and the Spike
        // reference build (VLEN=1024) grants half of what the VLEN=2048
        // bitstream does. A hardcoded expectation would fail for the wrong
        // reason on one of the two.
        RED_M8(  64, "T10 vl=64");
        RED_M8( 128, "T10 vl=128");
        RED_M8( 256, "T10 vl=256");
        RED_M8( 512, "T10 vl=512");
        RED_M8(1024, "T10 vl=1024");     // bn1's request; VLMAX on 4-lane HW
        // A non-power-of-two request, so the granted vl need not fill the
        // register group: if tail lanes are folded in rather than left out,
        // this is where it shows.
        RED_M8(1000, "T10 vl=1000");

        // LMUL sweep at a fixed 64 elements: isolates register-group width from
        // element count.
        printf_("\n[T11] vfredusum.vs LMUL sweep @ vl=64 (sum = 64.0)\n");
#define RED_LMUL(LM, nm)                                                       \
        do {                                                                   \
            asm volatile("vsetvli %0, %1, e16, " LM ", ta, ma"                  \
                         : "=r"(rvl) : "r"((long)64));                          \
            asm volatile("vle16.v v8, (%0)"  :: "r"(big));                      \
            asm volatile("vfmv.s.f v16, %0"  :: "f"(zero16));                   \
            asm volatile("vfredusum.vs v16, v8, v16");                          \
            asm volatile("vfmv.f.s %0, v16"  : "=f"(res));                      \
            chk(nm, bits16(res), 0x5400);                                       \
        } while (0)
        RED_LMUL("m1", "T11 m1 vl=64");
        RED_LMUL("m2", "T11 m2 vl=64");
        RED_LMUL("m4", "T11 m4 vl=64");
        RED_LMUL("m8", "T11 m8 vl=64");

        // T12: bn1's exact prologue -- `vsetvli zero` (= VLMAX) BEFORE the
        // vfmv.s.f seed, then a second vsetvli inside the loop. T8c set vl once
        // and seeded after; if the accumulator's tail is disturbed by the
        // intervening vsetvli, only this ordering shows it.
        // T12 reproduces bn1's loop EXACTLY, including the `vsetvli zero`
        // (= VLMAX) that precedes the vfmv.s.f seed and the re-vsetvli inside
        // the loop. Because it loops until all 1024 elements are consumed, the
        // expected total is 1024.0 whatever VLMAX is -- so this is a valid
        // comparison between the VLEN=1024 Spike build and the VLEN=2048
        // bitstream, where the hardware does it in ONE pass and Spike in two.
        printf_("\n[T12] bn1's exact loop over 1024 elements (sum = 1024.0)\n");
        {
            unsigned long vlmax;
            const _Float16 *p = big;
            long s = NBIG;
            asm volatile("vsetvli %0, zero, e16, m8, ta, ma" : "=r"(vlmax));
            printf_("    (vlmax @ e16,m8 = %lu -> %s)\n", vlmax,
                    vlmax >= NBIG ? "single pass, as on 4-lane HW"
                                  : "multi pass");
            asm volatile("vfmv.s.f v16, %0" :: "f"(zero16));
            while (s > 0) {
                asm volatile("vsetvli %0, %1, e16, m8, ta, ma"
                             : "=r"(rvl) : "r"(s));
                asm volatile("vle16.v v8, (%0)" :: "r"(p));
                asm volatile("vfredusum.vs v16, v8, v16");
                p += rvl;
                s -= (long)rvl;
            }
            asm volatile("vfmv.f.s %0, v16" : "=f"(res));
            chk("T12 bn1 loop 1024x1.0", bits16(res), 0x6400);
        }

        // T13: same loop, mixed data. All-ones can mask a data-dependent fault;
        // relu1's output is non-negative with many exact zeros, so use that
        // shape. 512 ones + 512 zeros = 512.0.
        printf_("\n[T13] bn1 loop, zeros interleaved (sum = 512.0)\n");
        {
            const _Float16 *p = big;
            long s = NBIG;
            for (int i = 0; i < NBIG; i++)
                big[i] = (i & 1) ? (_Float16)0.0f : (_Float16)1.0f;
            asm volatile("vsetvli %0, zero, e16, m8, ta, ma" : "=r"(rvl));
            asm volatile("vfmv.s.f v16, %0" :: "f"(zero16));
            while (s > 0) {
                asm volatile("vsetvli %0, %1, e16, m8, ta, ma"
                             : "=r"(rvl) : "r"(s));
                asm volatile("vle16.v v8, (%0)" :: "r"(p));
                asm volatile("vfredusum.vs v16, v8, v16");
                p += rvl;
                s -= (long)rvl;
            }
            asm volatile("vfmv.f.s %0, v16" : "=f"(res));
            chk("T13 bn1 loop mixed", bits16(res), 0x6000);
        }
    }

    // ---- T14: the DIVSQRT structural hazard --------------------------------
    // T7a fails on `x = fsqrt.s(1e-5); 1.0f/x` while T9 shows fdiv.s exact for
    // every constant divisor INCLUDING that exact value, and fsqrt.s exact on
    // its own. Both T7a operations are inline asm, so a compiler rsqrt lowering
    // is excluded by construction. In the object code the failing pair is only
    // four instructions apart, while the PASSING pair has a whole function call
    // between them -- and the intervening fmv.x.w reads the sqrt result
    // correctly, so the register dependency was tracked. That points at the
    // shared MERGED DIVSQRT unit (cva6 fpu_wrap.sv: '{default: MERGED}) not
    // being ready to accept a second operation.
    //
    // These tests vary the ISSUE GAP and the DEPENDENCE separately, which
    // separates "unit still busy" from "sqrt->div result forwarding".
    printf_("\n[T14] DIVSQRT issue-gap hazard\n");
    {
        volatile float epsv = 1e-5f, onev = 1.0f;
        float e = epsv, o = onev, r;

        // Gap sweep, dependent: fsqrt then fdiv by its result, N nops apart.
        // If this is a structural hazard there is a minimum N that works, and
        // that N is the size of the hazard window.
#define DIVSQRT_GAP(NOPS, nm)                                                  \
        do {                                                                   \
            asm volatile("fsqrt.s %0, %1\n\t"                                   \
                         ".rept " #NOPS "\n\tnop\n\t.endr\n\t"                  \
                         "fdiv.s %0, %2, %0"                                    \
                         : "=&f"(r) : "f"(e), "f"(o));                          \
            chk(nm, bits32(r), 0x439e1d28u);                                    \
        } while (0)
        DIVSQRT_GAP(0,  "T14a gap=0  nops");
        DIVSQRT_GAP(1,  "T14a gap=1  nop");
        DIVSQRT_GAP(2,  "T14a gap=2  nops");
        DIVSQRT_GAP(4,  "T14a gap=4  nops");
        DIVSQRT_GAP(8,  "T14a gap=8  nops");
        DIVSQRT_GAP(16, "T14a gap=16 nops");
        DIVSQRT_GAP(32, "T14a gap=32 nops");
        DIVSQRT_GAP(64, "T14a gap=64 nops");

        // T14b: fdiv INDEPENDENT of the sqrt, issued immediately after it.
        // Still contends for the unit, but needs no result forwarding.
        // Fails => structural (unit busy). Passes => forwarding-specific.
        {
            float q, s;
            volatile float fourv = 4.0f;
            float f = fourv;
            asm volatile("fsqrt.s %0, %2\n\t"
                         "fdiv.s  %1, %3, %4"
                         : "=&f"(s), "=&f"(q) : "f"(e), "f"(o), "f"(f));
            chk("T14b indep fdiv after fsqrt", bits32(q), 0x3e800000u); // 1/4
            chk("T14b  ... the fsqrt result",  bits32(s), 0x3b4f3e37u);
        }

        // T14c: two dependent fdiv.s back to back, no sqrt involved. Isolates
        // "any back-to-back DIVSQRT op" from "sqrt followed by div".
        {
            float q;
            volatile float twov = 2.0f;
            float t = twov;
            asm volatile("fdiv.s %0, %1, %2\n\t"      // 1/2   = 0.5
                         "fdiv.s %0, %1, %0"          // 1/0.5 = 2.0
                         : "=&f"(q) : "f"(o), "f"(t));
            chk("T14c fdiv->fdiv dependent", bits32(q), 0x40000000u);
        }

        // T14d: two dependent fsqrt.s back to back.
        {
            float s;
            volatile float sixteen = 16.0f;
            float f = sixteen;
            asm volatile("fsqrt.s %0, %1\n\t"         // 4.0
                         "fsqrt.s %0, %0"             // 2.0
                         : "=&f"(s) : "f"(f));
            chk("T14d fsqrt->fsqrt dependent", bits32(s), 0x40000000u);
        }
    }

    // ---- T15: integer -> float conversion ---------------------------------
    // The old T10 harness computed its expectation as `(_Float16)(float)vl` and
    // got 0x7c00 (+Inf) for EVERY vl on hardware, while Spike computed it
    // correctly. That expectation is now integer-only (u2f16_bits), but the
    // conversion itself is worth testing: it is `fcvt.s.lu`/`fcvt.s.l`, which
    // nothing has covered, and 0x7c00 says it returns something enormous.
    printf_("\n[T15] integer -> float conversion (fcvt.s.lu / fcvt.s.l)\n");
    {
        volatile unsigned long uv = 64;
        volatile long         sv = 64;
        float fu, fs;
        asm volatile("fcvt.s.lu %0, %1" : "=f"(fu) : "r"(uv));
        asm volatile("fcvt.s.l  %0, %1" : "=f"(fs) : "r"(sv));
        chk("T15 fcvt.s.lu(64)", bits32(fu), 0x42800000u);      // 64.0f
        chk("T15 fcvt.s.l (64)", bits32(fs), 0x42800000u);
        // and the full chain the old harness used
        volatile _Float16 h = (_Float16)(float)uv;
        chk("T15 (f16)(float)64ul", bits16(h), 0x5400u);        // 64.0
        volatile unsigned long uv2 = 1024;
        volatile _Float16 h2 = (_Float16)(float)uv2;
        chk("T15 (f16)(float)1024ul", bits16(h2), 0x6400u);     // 1024.0
    }

    // ---- T16: the exact T7a triple ---------------------------------------
    // T14a proved issue DISTANCE is not the trigger: even gap=0 (dependent
    // fsqrt.s immediately followed by fdiv.s) is correct. The one structural
    // difference in the failing T7a code is an `fmv.x.w` that READS the
    // fsqrt result register between the producer and the dependent fdiv:
    //
    //     fsqrt.s  fa5, fa5
    //     fmv.x.w  a3, fa5      <- GPR read of the FP result, mid-dependency
    //     fdiv.s   fa5, fa4, fa5
    //
    // T16a reproduces exactly that. T16b/T16c vary it to say whether the
    // FP->GPR read is what matters or merely any intervening consumer.
    printf_("\n[T16] fsqrt -> fmv.x.w -> dependent fdiv\n");
    {
        volatile float epsv = 1e-5f, onev = 1.0f;
        float e = epsv, o = onev, q;
        unsigned mid;

        // T16a: the exact failing triple.
        asm volatile("fsqrt.s %0, %2\n\t"
                     "fmv.x.w %1, %0\n\t"
                     "fdiv.s  %0, %3, %0"
                     : "=&f"(q), "=&r"(mid) : "f"(e), "f"(o));
        chk("T16a sqrt bits via fmv.x.w", mid, 0x3b4f3e37u);
        chk("T16a dependent fdiv result", bits32(q), 0x439e1d28u);

        // T16b: same, but the fmv.x.w reads a DIFFERENT f-register, so the
        // FP->GPR move still happens but not on the dependency chain.
        {
            float q2; unsigned m2;
            asm volatile("fsqrt.s %0, %2\n\t"
                         "fmv.x.w %1, %3\n\t"
                         "fdiv.s  %0, %3, %0"
                         : "=&f"(q2), "=&r"(m2) : "f"(e), "f"(o));
            chk("T16b fmv.x.w of other reg", bits32(q2), 0x439e1d28u);
        }

        // T16c: an FP consumer (not FP->GPR) between producer and consumer.
        {
            float q3, t3;
            asm volatile("fsqrt.s %0, %3\n\t"
                         "fadd.s  %1, %0, %0\n\t"
                         "fdiv.s  %0, %4, %0"
                         : "=&f"(q3), "=&f"(t3) : "f"(e), "f"(e), "f"(o));
            chk("T16c fadd.s between", bits32(q3), 0x439e1d28u);
        }
    }

    // ---- T17: .vf scalar operand held across intervening instructions -----
    // T2 (`vfmacc.vf`) regressed to 0x7e00 -- the dispatcher's NaN-box
    // substitution -- while T5, which uses the SAME instruction, passes. In the
    // object code T2's scalar is loaded by `flh fa5` and then used four
    // instructions later, with two `vfmv.v.f` in between. `flh` NaN-boxes by
    // definition and T8d proved flh boxing works here, so a 0x7e00 means the
    // dispatcher saw an unboxed (likely zero) operand -- i.e. the f-register
    // read back wrong at issue time. Sweep the distance between loading the
    // scalar and consuming it.
    printf_("\n[T17] .vf scalar held across N intervening vector ops\n");
    {
        _Float16 sc_one = (_Float16)1.0f, sc_two = (_Float16)2.0f,
                 sc_three = (_Float16)3.0f;
        asm volatile("vsetvli t0, %0, e16, m1, ta, ma" :: "r"((long)N) : "t0");

        // T17a: scalar consumed IMMEDIATELY (T5's shape -- known good).
        asm volatile("vfmv.v.f v8, %0"  :: "f"(sc_two));
        asm volatile("vfmv.v.f v16, %0" :: "f"(sc_one));
        asm volatile("vfmacc.vf v16, %0, v8" :: "f"(sc_three));
        asm volatile("vse16.v v16, (%0)" :: "r"(dst));
        chk("T17a scalar used immediately", bits16(dst[0]), 0x4700);

        // T17b: the scalar is materialised FIRST, then two vector ops issue
        // before it is consumed -- T2's exact ordering.
        {
            register _Float16 held asm("fa5") = sc_three;
            asm volatile("" :: "f"(held));           // pin it before the loads
            asm volatile("vfmv.v.f v8, %0"  :: "f"(sc_two));
            asm volatile("vfmv.v.f v16, %0" :: "f"(sc_one));
            asm volatile("vfmacc.vf v16, %0, v8" :: "f"(held));
            asm volatile("vse16.v v16, (%0)" :: "r"(dst));
            chk("T17b scalar held across 2 ops", bits16(dst[0]), 0x4700);
        }

        // T17c: same, but with the scalar reloaded right before the use.
        {
            _Float16 fresh = sc_three;
            asm volatile("vfmv.v.f v8, %0"  :: "f"(sc_two));
            asm volatile("vfmv.v.f v16, %0" :: "f"(sc_one));
            asm volatile("" : "+f"(fresh));          // force a fresh definition
            asm volatile("vfmacc.vf v16, %0, v8" :: "f"(fresh));
            asm volatile("vse16.v v16, (%0)" :: "r"(dst));
            chk("T17c scalar refreshed at use", bits16(dst[0]), 0x4700);
        }
    }

    // ---- T18: BACK-TO-BACK reductions, as bn1 actually does them ----------
    // Every reduction test so far (T8c, T10-T13) has a printf_/chk call
    // between reductions -- thousands of cycles of unrelated work. bn1 runs
    // 384 reductions (32 images x 12 channels) back to back with nothing but
    // pointer arithmetic in between, and it is the FIRST reduction anywhere in
    // the network (conv is vfmacc.vf, relu is vmax). If the reduction FSM
    // (INTRA_LANE -> INTER_LANES TX/RX -> LN0_COMMIT -> SIMD, with the
    // red_hs_synch / sldu_transactions_cnt handshake) does not fully settle
    // between consecutive reductions, the 2nd and later ones corrupt while an
    // isolated one is always perfect. That matches everything observed.
    //
    // Results are stored, NOT printed, inside the loop -- printing would
    // reintroduce exactly the spacing this test exists to remove.
    printf_("\n[T18] back-to-back vfredusum.vs (bn1's real pattern)\n");
    {
        enum { NRED = 64 };
        static _Float16 red_res[NRED];
        _Float16 zero16 = (_Float16)0.0f;
        unsigned long rvl;

        for (int i = 0; i < NBIG; i++) big[i] = (_Float16)1.0f;
        for (int i = 0; i < NRED; i++) red_res[i] = (_Float16)0.0f;

        // Tight loop: seed, reduce 256 elements, extract. No calls between.
        for (int k = 0; k < NRED; k++) {
            _Float16 acc;
            asm volatile("vsetvli %0, %1, e16, m8, ta, ma"
                         : "=r"(rvl) : "r"((long)256));
            asm volatile("vle16.v v8, (%0)"  :: "r"(big));
            asm volatile("vfmv.s.f v16, %0"  :: "f"(zero16));
            asm volatile("vfredusum.vs v16, v8, v16");
            asm volatile("vfmv.f.s %0, v16"  : "=f"(acc));
            red_res[k] = acc;
        }

        // Verify afterwards. Report the FIRST bad index: if reduction 0 is
        // right and some later one is wrong, that index is the finding.
        {
            int firstbad = -1, nbad = 0;
            for (int k = 0; k < NRED; k++) {
                if (bits16(red_res[k]) != 0x5c00) {      // 256.0
                    nbad++;
                    if (firstbad < 0) firstbad = k;
                }
            }
            printf_("    red[0]=0x%04x red[1]=0x%04x red[2]=0x%04x red[63]=0x%04x\n",
                    bits16(red_res[0]), bits16(red_res[1]),
                    bits16(red_res[2]), bits16(red_res[NRED-1]));
            printf_("    bad=%d of %d, first bad index=%d\n", nbad, (int)NRED, firstbad);
            chk("T18a all 64 reductions", (unsigned)nbad, 0u);
            chk("T18a first reduction",   bits16(red_res[0]), 0x5c00);
            chk("T18a second reduction",  bits16(red_res[1]), 0x5c00);
        }

        // T18b: bn1's exact two-level shape -- an OUTER accumulate over
        // several reductions, like `op->avg[c] += sum` across 32 images.
        {
            _Float16 total = (_Float16)0.0f;
            for (int k = 0; k < 32; k++) {
                _Float16 acc;
                asm volatile("vsetvli %0, %1, e16, m8, ta, ma"
                             : "=r"(rvl) : "r"((long)256));
                asm volatile("vle16.v v8, (%0)"  :: "r"(big));
                asm volatile("vfmv.s.f v16, %0"  :: "f"(zero16));
                asm volatile("vfredusum.vs v16, v8, v16");
                asm volatile("vfmv.f.s %0, v16"  : "=f"(acc));
                total = (_Float16)(total + acc);          // 32 * 256 = 8192
            }
            chk("T18b 32x accumulated", bits16(total), 0x7000);   // 8192.0
        }
    }

    // ---- T19: map which .rodata addresses refill wrong after an eviction ---
    // ESTABLISHED (hardware, ARM-side devmem with CVA6 in reset): DRAM holds
    // the fp16 constant 1.0 correctly at 0x40004F60 BOTH BEFORE AND AFTER a
    // run, yet CVA6 reads it as 0x0000. So CVA6's LOAD returns zero from
    // correct memory -- a cold-refill fault on a line CVA6 never wrote.
    //
    // That address is (a) the first halfword of .rodata, (b) the first halfword
    // of a 32-byte D-cache line (DcacheLineWidth=256, and it is 32B aligned),
    // and (c) the line immediately after the last line of .text. Three
    // coincident properties, and the failure is NOT "every line's first
    // halfword" -- other .rodata constants further on read correctly.
    //
    // This probe decides between them. probe_ro[] is a known 1 KiB pattern in
    // .rodata. We first evict the whole LLC by rewriting the 1 MiB sweep
    // buffer, so every probe_ro read below is a genuine cold refill, then read
    // all 512 entries and report which ones come back wrong -- by index, by
    // address, and by offset within their 32-byte line.
    printf_("\n[T19] .rodata cold-refill map\n");
    {
        const unsigned long base = (unsigned long)(unsigned long long)&probe_ro[0];
        printf_("    probe_ro base=0x%08lx  (32B-aligned: %s)\n",
                base, (base & 31UL) ? "no" : "YES");

        // Evict: rewrite the 1 MiB sweep buffer so nothing of probe_ro can
        // still be cached anywhere.
        for (unsigned long i = 0; i < SWEEP_HW; i++)
            sweep_buf[i] = (unsigned short)(i & 0xFFFFUL);

        int bad = 0, firstbad = -1;
        int bad_at_off0 = 0, bad_elsewhere = 0;
        for (int i = 0; i < 512; i++) {
            unsigned short g = *(const volatile unsigned short *)&probe_ro[i];
            unsigned short e = (unsigned short)((i * 3 + 0x1001) & 0xFFFF);
            if (g != e) {
                unsigned long a = base + 2UL * (unsigned long)i;
                if (bad == 0) firstbad = i;
                if ((a & 31UL) == 0) bad_at_off0++; else bad_elsewhere++;
                if (bad < 8)
                    printf_("    bad i=%-4d addr=0x%08lx off-in-line=%-2lu "
                            "got=0x%04x exp=0x%04x\n",
                            i, a, (a & 31UL), (unsigned)g, (unsigned)e);
                bad++;
            }
        }
        printf_("    bad=%d of 512   first=%d   at-line-offset-0=%d  elsewhere=%d\n",
                bad, firstbad, bad_at_off0, bad_elsewhere);
        chk("T19 rodata cold refill", (unsigned)bad, 0u);
    }

    // ---- T20: is the trigger `vsetvli` immediately before a .vf op? -------
    // T17a fails while T17b/T17c pass, yet the object code shows ALL THREE have
    // the producing f-register written by the IMMEDIATELY preceding instruction
    // -- so "zero gap between the f-reg write and the .vf use" is NOT the
    // trigger. T5 is zero-gap too (fmv.h.x fa5,s0 ; vfmacc.vf v16,fa5,v24) and
    // passes.
    //
    // What actually separates them is a `vsetvli` shortly before the .vf op:
    //   T17a  vsetvli e16,m1 ... vfmacc.vf        -> FAIL
    //   T17b/c  no vsetvli, vtype inherited       -> PASS
    //   T2/T5   no vsetvli                        -> PASS
    //   T8c/T8d/T18a  vsetvli, but a vle16.v between -> PASS
    //   T12     vsetvli, but a printf call between   -> PASS
    // and bn1, whose channel 0 is the ONLY failing channel, is exactly:
    //   vsetvli a2, zero, e16, m8 ; vfmv.s.f v16, fa3
    //
    // These four sub-tests confirm the rule and find the minimal workaround.
    // All operands identical throughout; only the instruction between the
    // vsetvli and the .vf changes.
    printf_("\n[T20] vsetvli immediately before a .vf scalar op\n");
    {
        _Float16 sc = (_Float16)1.0f;

        // T20a: vsetvli that CHANGES vtype, then .vf with nothing between.
        // This is bn1's exact shape. Expect FAIL if the rule holds.
        asm volatile("vsetvli t0, %0, e16, m8, ta, ma" :: "r"((long)N) : "t0");
        asm volatile("vsetvli t0, %0, e16, m1, ta, ma\n\t"
                     "vfmv.v.f v8, %1"
                     :: "r"((long)N), "f"(sc) : "t0");
        asm volatile("vse16.v v8, (%0)" :: "r"(dst));
        chk("T20a vsetvli->vf  (gap 0)", bits16(dst[0]), 0x3C00);

        // T20b: same, with ONE scalar nop between. Does a non-vector
        // instruction suffice, or must it be a vector op?
        asm volatile("vsetvli t0, %0, e16, m8, ta, ma" :: "r"((long)N) : "t0");
        asm volatile("vsetvli t0, %0, e16, m1, ta, ma\n\t"
                     "nop\n\t"
                     "vfmv.v.f v8, %1"
                     :: "r"((long)N), "f"(sc) : "t0");
        asm volatile("vse16.v v8, (%0)" :: "r"(dst));
        chk("T20b vsetvli->nop->vf", bits16(dst[0]), 0x3C00);

        // T20c: same, with an intervening VECTOR op (the shape that passes in
        // T8c/T18a).
        asm volatile("vsetvli t0, %0, e16, m8, ta, ma" :: "r"((long)N) : "t0");
        asm volatile("vsetvli t0, %0, e16, m1, ta, ma\n\t"
                     "vle16.v v24, (%2)\n\t"
                     "vfmv.v.f v8, %1"
                     :: "r"((long)N), "f"(sc), "r"(src) : "t0");
        asm volatile("vse16.v v8, (%0)" :: "r"(dst));
        chk("T20c vsetvli->vle16->vf", bits16(dst[0]), 0x3C00);

        // T20d: vsetvli that does NOT change vtype (already e16,m1), then .vf
        // with nothing between. bn1 fails only on channel 0 -- the one
        // iteration where the vsetvli is an actual vtype CHANGE -- so if the
        // change is what matters, this passes while T20a fails.
        asm volatile("vsetvli t0, %0, e16, m1, ta, ma" :: "r"((long)N) : "t0");
        asm volatile("vsetvli t0, %0, e16, m1, ta, ma\n\t"
                     "vfmv.v.f v8, %1"
                     :: "r"((long)N), "f"(sc) : "t0");
        asm volatile("vse16.v v8, (%0)" :: "r"(dst));
        chk("T20d redundant vsetvli->vf", bits16(dst[0]), 0x3C00);
    }

    // ---- T21: does `flh` differ from `lhu` on the SAME address? -----------
    // Every failure that MOVES between runs (T6c's 2.0, T11's seed, T17a, and
    // earlier `one`) has the same signature: a scalar f-register value arriving
    // as ZERO, or as 0x7e00 (the dispatcher substituting canonical NaN for an
    // operand it sees as unboxed -- which an all-zero f-reg is).
    //
    // Meanwhile T0 and T19 pass. But BOTH read memory with INTEGER loads:
    // T19 reads probe_ro via `*(const volatile unsigned short *)`, i.e. `lhu`.
    // Neither has ever exercised `flh`, the fp16 load that actually produces
    // those scalars. So "memory is fine" was only ever established for the
    // integer load path.
    //
    // T21 reads the SAME 512 addresses BOTH ways and compares. An `flh` that
    // returns 0 where `lhu` returns the correct value is a CVA6 FP-load-path
    // fault and would explain the whole moving-failure class at once.
    // The LLC is evicted first so every read is a cold refill.
    printf_("\n[T21] flh vs lhu on identical addresses\n");
    {
        const unsigned long base = (unsigned long)(unsigned long long)&probe_ro[0];
        printf_("    probe_ro base=0x%08lx\n", base);

        for (unsigned long i = 0; i < SWEEP_HW; i++)
            sweep_buf[i] = (unsigned short)(i & 0xFFFFUL);   // evict the LLC

        int bad = 0, firstbad = -1, flh_zero = 0;
        for (int i = 0; i < 512; i++) {
            unsigned short gi = *(const volatile unsigned short *)&probe_ro[i];
            _Float16 gf;
            asm volatile("flh %0, 0(%1)" : "=f"(gf) : "r"(&probe_ro[i]));
            unsigned gfb = bits16(gf);
            if ((unsigned)gi != gfb) {
                if (bad == 0) firstbad = i;
                if (gfb == 0) flh_zero++;
                if (bad < 8)
                    printf_("    i=%-4d addr=0x%08lx  lhu=0x%04x  flh=0x%04x\n",
                            i, base + 2UL * (unsigned long)i,
                            (unsigned)gi, gfb);
                bad++;
            }
        }
        printf_("    lhu/flh mismatches=%d of 512  first=%d  (flh read 0: %d)\n",
                bad, firstbad, flh_zero);
        chk("T21 flh == lhu", (unsigned)bad, 0u);

        // T21b: the same address read by flh TWICE in a row -- is a single
        // address even self-consistent?
        {
            _Float16 a1, a2;
            asm volatile("flh %0, 0(%1)" : "=f"(a1) : "r"(&probe_ro[0]));
            asm volatile("flh %0, 0(%1)" : "=f"(a2) : "r"(&probe_ro[0]));
            printf_("    probe_ro[0] flh twice: 0x%04x 0x%04x (exp 0x1001)\n",
                    bits16(a1), bits16(a2));
            chk("T21b flh repeatable", bits16(a1), bits16(a2));
        }
    }

    printf_("\n===== vec_test done: %d failure(s) =====\n", fails);
    return fails;
}
