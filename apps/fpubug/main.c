// fpubug - minimal, self-contained repros for the four defects that survived
// every build and every DRAM history in the vec_test A/B/C run matrix (§36).
//
// Design rules, chosen because they remove the confounds that wasted whole
// rounds on vec_test:
//   1. EVERY operand is built with fmv.w.x / fmv.h.x from an immediate.
//      Nothing is loaded from memory, so the entire stale-cold-refill class
//      (§26-§33) cannot reach these tests.
//   2. Each test is a single asm volatile block with explicit registers, so
//      the compiler cannot reschedule across the sequence under test. Adding
//      a test cannot move another one's schedule (the §35/§36 cross-build
//      drift).
//   3. Every failing shape is paired with a CONTROL that differs in exactly
//      one respect, so a FAIL/PASS pair names the mechanism by itself.
//
// Build: -DFPGA=1 -DNO_BOOT_LLC_FLUSH=1

#include <stdint.h>
#include "printf.h"

static int fails = 0;
static void chk32(const char *nm, unsigned got, unsigned exp) {
    int ok = (got == exp);
    if (!ok) fails++;
    printf_("  [%s] %-30s got=0x%08x exp=0x%08x\n", ok ? "PASS" : "FAIL", nm, got, exp);
}
static void chk16(const char *nm, unsigned got, unsigned exp) {
    int ok = (got == exp);
    if (!ok) fails++;
    printf_("  [%s] %-30s got=0x%04x exp=0x%04x\n", ok ? "PASS" : "FAIL", nm, got, exp);
}

// Read a scalar f-register's raw FLEN=32 bit pattern without going near memory.
#define FBITS(dst, freg) asm volatile("fmv.x.w %0, " freg : "=r"(dst))

int main(void) {
    printf_("\n===== fpubug: minimal repros (no memory operands) =====\n");

    // =================================================================
    // R1 - fcvt.s.l / fcvt.s.lu  (vec_test T15)
    //
    // STATIC ROOT CAUSE (cva6_patched + cvfpu, read 2026-09-01):
    //   fpu_wrap.sv:64      Width      = CVA6Cfg.FLen = 32   (RVD=0)
    //   fpu_wrap.sv:68-73   IntFmtMask = {.., INT32:1, INT64:1}  <-- INT64 asked for
    //   fpu_wrap.sv:227,246 fcvt.s.l/.lu request fpu_ifmt = INT64
    //   fpnew_pkg.sv:441    get_conv_lane_int_formats() enables an INT format
    //                       ONLY if an FP format of the SAME WIDTH is active.
    //                       INT64 (64b) therefore needs FP64, and FP64 = RVD = 0.
    //                       => INT64 is masked OFF in IntFmtConfig.
    //   fpnew_cast_multi.sv:239-240  inactive format drives
    //                       ifmt_input_val[ifmt] = '{default: DONT_CARE}
    //   fpnew_pkg.sv:287    DONT_CARE = 1'b1
    //   fpnew_cast_multi.sv:245      int_value = ifmt_input_val[int_fmt_q]
    //
    //   So int_value is ALL ONES at MAX_INT_WIDTH = 32, and the OPERAND IS
    //   NEVER READ. Predicted results, independent of the input value:
    //       fcvt.s.lu -> 0xFFFFFFFF unsigned = 4294967295 -> 0x4F800000 (2^32)
    //       fcvt.s.l  -> 0xFFFFFFFF signed   = -1         -> 0xBF800000
    //   Both match the observed T15 words exactly.
    //
    // DISCRIMINATOR: sweep the input. If the theory is right, every input
    // gives the SAME wrong answer. If the wrong answers track the input, the
    // theory is wrong and the operand is reaching the unit after all.
    // =================================================================
    printf_("\n[R1] fcvt int64<->float  (predict: result INDEPENDENT of operand)\n");
    {
        static const unsigned long inputs[4] = { 64, 7, 1, 0 };
        for (int i = 0; i < 4; i++) {
            unsigned long v = inputs[i];
            unsigned bu, bs, bwu, bws;
            asm volatile("fcvt.s.lu ft0, %1\n\tfmv.x.w %0, ft0"
                         : "=r"(bu) : "r"(v) : "ft0");
            asm volatile("fcvt.s.l  ft0, %1\n\tfmv.x.w %0, ft0"
                         : "=r"(bs) : "r"(v) : "ft0");
            asm volatile("fcvt.s.wu ft0, %1\n\tfmv.x.w %0, ft0"
                         : "=r"(bwu) : "r"(v) : "ft0");
            asm volatile("fcvt.s.w  ft0, %1\n\tfmv.x.w %0, ft0"
                         : "=r"(bws) : "r"(v) : "ft0");
            printf_("    in=%2lu  s.lu=0x%08x  s.l=0x%08x  |  s.wu=0x%08x  s.w=0x%08x\n",
                    v, bu, bs, bwu, bws);
        }
        printf_("    ^ if the s.lu / s.l columns are CONSTANT across all four\n"
                "      inputs, the INT64 DONT_CARE path is confirmed.\n");
        // The table above is printed but was NOT counted -- so a clean summary
        // line could hide four real failures. Count them explicitly.
        {
            unsigned e64[4] = { 0x42800000u, 0x40e00000u, 0x3f800000u, 0x00000000u };
            int badlu = 0, badl = 0;
            for (int i = 0; i < 4; i++) {
                unsigned long v = inputs[i]; unsigned bu, bs;
                asm volatile("fcvt.s.lu ft0, %1\n\tfmv.x.w %0, ft0"
                             : "=r"(bu) : "r"(v) : "ft0");
                asm volatile("fcvt.s.l  ft0, %1\n\tfmv.x.w %0, ft0"
                             : "=r"(bs) : "r"(v) : "ft0");
                if (bu != e64[i]) badlu++;
                if (bs != e64[i]) badl++;
            }
            chk32("R1 fcvt.s.lu all 4 inputs", (unsigned)badlu, 0u);
            chk32("R1 fcvt.s.l  all 4 inputs", (unsigned)badl,  0u);
        }
        // 64.0f == 0x42800000. The .w/.wu forms use INT32 and must pass.
        unsigned c1, c2;
        asm volatile("li t1, 64\n\tfcvt.s.wu ft0, t1\n\tfmv.x.w %0, ft0"
                     : "=r"(c1) :: "t1", "ft0");
        asm volatile("li t1, 64\n\tfcvt.s.w  ft0, t1\n\tfmv.x.w %0, ft0"
                     : "=r"(c2) :: "t1", "ft0");
        chk32("R1 CONTROL fcvt.s.wu(64)", c1, 0x42800000u);
        chk32("R1 CONTROL fcvt.s.w (64)", c2, 0x42800000u);
    }

    // =================================================================
    // R2 - fsqrt.s -> dependent fdiv.s  (vec_test T7a)
    //
    // OBSERVED: fsqrt.s(16.0) PASSES, fdiv.s(1,4) PASSES, sqrt(1e-5) PASSES,
    // but 1.0/sqrt(1e-5) returns 0x00000000 = +0.0.
    //
    // In vec_test the PASSING divide has a chk()/printf_ (thousands of cycles)
    // between the fsqrt and the fdiv; the FAILING one consumes the fsqrt
    // result immediately. cvfpu's DIVSQRT is MERGED (fpu_wrap.sv:91) onto the
    // iterative div_sqrt_top_mvp (fpnew_divsqrt_multi.sv:278), whose FSM
    // accepts a NEW op in the SAME cycle it hands the previous result
    // downstream (fpnew_divsqrt_multi.sv:216-222: out_ready -> in_ready=1,
    // state stays BUSY). NOTE: vec_test T14 swept 0-64 NOPS between two
    // divsqrt ops and passed -- but NOPS do not create a data dependency, so
    // T14 never actually built this shape. That is why it "refuted" nothing.
    //
    // DISCRIMINATOR: R2c preloads the sqrt RESULT as an immediate bit pattern
    // and divides with no fsqrt anywhere in front of it. Same operands, no
    // preceding divsqrt op.
    //   R2a FAIL + R2c PASS -> the fault is fsqrt->fdiv SEQUENCING.
    //   R2a FAIL + R2c FAIL -> the fault is fdiv on THESE OPERANDS.
    // =================================================================
    printf_("\n[R2] fsqrt.s -> dependent fdiv.s\n");
    {
        const unsigned EPS  = 0x3727c5acu;  // 1e-5f
        const unsigned ONE  = 0x3f800000u;  // 1.0f
        const unsigned RT   = 0x3b4f3e37u;  // sqrtf(1e-5f), from the Spike run
        const unsigned EXP  = 0x439e1d28u;  // 1.0f/sqrtf(1e-5f), from Spike
        unsigned g;

        // R2a: dependent, zero gap. This is the failing shape.
        asm volatile("fmv.w.x ft0, %1\n\t"
                     "fmv.w.x ft1, %2\n\t"
                     "fsqrt.s ft2, ft0\n\t"
                     "fdiv.s  ft3, ft1, ft2\n\t"
                     "fmv.x.w %0, ft3"
                     : "=r"(g) : "r"(EPS), "r"(ONE) : "ft0","ft1","ft2","ft3");
        chk32("R2a fsqrt->fdiv gap=0", g, EXP);

        // R2b: same, 8 nops between the producer and the consumer.
        asm volatile("fmv.w.x ft0, %1\n\t"
                     "fmv.w.x ft1, %2\n\t"
                     "fsqrt.s ft2, ft0\n\t"
                     ".rept 8\n\tnop\n\t.endr\n\t"
                     "fdiv.s  ft3, ft1, ft2\n\t"
                     "fmv.x.w %0, ft3"
                     : "=r"(g) : "r"(EPS), "r"(ONE) : "ft0","ft1","ft2","ft3");
        chk32("R2b fsqrt->fdiv gap=8", g, EXP);

        // R2c: CONTROL. The divisor is the sqrt result as a literal; there is
        // no fsqrt in front of the fdiv at all.
        asm volatile("fmv.w.x ft1, %1\n\t"
                     "fmv.w.x ft2, %2\n\t"
                     "fdiv.s  ft3, ft1, ft2\n\t"
                     "fmv.x.w %0, ft3"
                     : "=r"(g) : "r"(ONE), "r"(RT) : "ft1","ft2","ft3");
        chk32("R2c CONTROL fdiv, no fsqrt", g, EXP);

        // R2d: does the fsqrt itself still produce the right value in this
        // exact block? Reads ft2 back before the divide consumes it.
        unsigned gs;
        asm volatile("fmv.w.x ft0, %1\n\t"
                     "fsqrt.s ft2, ft0\n\t"
                     "fmv.x.w %0, ft2"
                     : "=r"(gs) : "r"(EPS) : "ft0","ft2");
        chk32("R2d fsqrt.s(1e-5) alone", gs, RT);

        // R2e: dependent fsqrt->fdiv but with the numerator materialised
        // AFTER the fsqrt, so the fdiv issues one cycle later.
        asm volatile("fmv.w.x ft0, %1\n\t"
                     "fsqrt.s ft2, ft0\n\t"
                     "fmv.w.x ft1, %2\n\t"
                     "fdiv.s  ft3, ft1, ft2\n\t"
                     "fmv.x.w %0, ft3"
                     : "=r"(g) : "r"(EPS), "r"(ONE) : "ft0","ft1","ft2","ft3");
        chk32("R2e fsqrt, fmv, fdiv", g, EXP);

        // R2f: two dependent divides in a row, no sqrt (is it divsqrt-unit
        // back-to-back in general, or fsqrt->fdiv specifically?).
        asm volatile("fmv.w.x ft1, %1\n\t"
                     "fmv.w.x ft2, %2\n\t"
                     "fdiv.s  ft3, ft1, ft2\n\t"
                     "fdiv.s  ft4, ft1, ft3\n\t"
                     "fmv.x.w %0, ft4"
                     : "=r"(g) : "r"(ONE), "r"(RT) : "ft1","ft2","ft3","ft4");
        chk32("R2f fdiv->dependent fdiv", g, 0x3b4f3e36u); // 1/(1/x), 1 ULP off RT -- value from the Spike run, not derived
    }

    // =================================================================
    // R3 - loop-carried scalar fp16 accumulate  (vec_test T18b)
    //
    // OBSERVED: 32 iterations of `total += vfredusum(...)` gives 0x5c00
    // (=256.0, exactly ONE term) instead of 0x7000 (=8192.0). T18a already
    // proved 64 back-to-back reductions are all individually correct, so the
    // reductions are NOT the problem -- the scalar accumulate is.
    //
    // R3a removes Ara from the picture entirely: the same 32-iteration
    // fp16 accumulate with a constant addend and no vector instruction
    // anywhere. If R3a fails, this is a pure CVA6 scalar bug and the vector
    // unit is exonerated.
    // =================================================================
    printf_("\n[R3] loop-carried scalar fp16 accumulate\n");
    {
        // NOTE: -march=rv64imac_zve64f_zvfh gives Zfhmin only -- fmv.h.x /
        // fcvt.h.s / fcvt.s.h exist, fadd.h does NOT. So a _Float16 add is
        // really fcvt.s.h, fcvt.s.h, fadd.s, fcvt.h.s, exactly as written
        // below. This IS what T18b's `total = (_Float16)(total + acc)`
        // compiles to, not an approximation of it.
        // 256.0h = 0x5c00, 32 * 256 = 8192.0h = 0x7000
        unsigned g;

        // R3a: 32-iteration accumulate, branch loop, no vector instruction
        // anywhere. If this fails, Ara is exonerated and the bug is a pure
        // CVA6 scalar defect.
        asm volatile("li   t1, 0x5c00\n\t"
                     "fmv.h.x ft1, t1\n\t"
                     "fmv.h.x ft0, zero\n\t"
                     "li   t2, 32\n\t"
                     "1:\n\t"
                     "fcvt.s.h ft2, ft0\n\t"
                     "fcvt.s.h ft3, ft1\n\t"
                     "fadd.s   ft2, ft2, ft3\n\t"
                     "fcvt.h.s ft0, ft2\n\t"
                     "addi t2, t2, -1\n\t"
                     "bnez t2, 1b\n\t"
                     "fmv.x.h %0, ft0"
                     : "=r"(g) :: "t1","t2","ft0","ft1","ft2","ft3");
        chk16("R3a 32x accum, no vector", (unsigned)(g & 0xFFFFu), 0x7000u);

        // R3b: same, unrolled -- is it the loop or the accumulate?
        asm volatile("li   t1, 0x5c00\n\t"
                     "fmv.h.x ft1, t1\n\t"
                     "fmv.h.x ft0, zero\n\t"
                     ".rept 32\n\t"
                     "fcvt.s.h ft2, ft0\n\t"
                     "fcvt.s.h ft3, ft1\n\t"
                     "fadd.s   ft2, ft2, ft3\n\t"
                     "fcvt.h.s ft0, ft2\n\t"
                     ".endr\n\t"
                     "fmv.x.h %0, ft0"
                     : "=r"(g) :: "t1","ft0","ft1","ft2","ft3");
        chk16("R3b 32x unrolled", (unsigned)(g & 0xFFFFu), 0x7000u);

        // R3c: only 2 accumulates -- is the very first one already wrong?
        asm volatile("li   t1, 0x5c00\n\t"
                     "fmv.h.x ft1, t1\n\t"
                     "fmv.h.x ft0, zero\n\t"
                     ".rept 2\n\t"
                     "fcvt.s.h ft2, ft0\n\t"
                     "fcvt.s.h ft3, ft1\n\t"
                     "fadd.s   ft2, ft2, ft3\n\t"
                     "fcvt.h.s ft0, ft2\n\t"
                     ".endr\n\t"
                     "fmv.x.h %0, ft0"
                     : "=r"(g) :: "t1","ft0","ft1","ft2","ft3");
        chk16("R3c 2x accum", (unsigned)(g & 0xFFFFu), 0x6000u);   // 512.0

        // R3d: CONTROL -- the identical accumulate kept entirely in fp32,
        // with no fp16 round trip. Isolates fcvt.h.s/fcvt.s.h from fadd.s.
        unsigned g32;
        asm volatile("li   t1, 0x43800000\n\t"      // 256.0f
                     "fmv.w.x ft1, t1\n\t"
                     "fmv.w.x ft0, zero\n\t"
                     ".rept 32\n\tfadd.s ft0, ft0, ft1\n\t.endr\n\t"
                     "fmv.x.w %0, ft0"
                     : "=r"(g32) :: "t1","ft0","ft1");
        chk32("R3d CONTROL 32x fadd.s", g32, 0x46000000u);         // 8192.0f

        // R3e: the fp16 round trip on its own, once -- does fcvt.h.s(fcvt.s.h)
        // preserve a value at all?
        unsigned rt;
        asm volatile("li   t1, 0x5c00\n\t"
                     "fmv.h.x ft1, t1\n\t"
                     "fcvt.s.h ft2, ft1\n\t"
                     "fcvt.h.s ft0, ft2\n\t"
                     "fmv.x.h %0, ft0"
                     : "=r"(rt) :: "t1","ft0","ft1","ft2");
        chk16("R3e fp16 round trip", (unsigned)(rt & 0xFFFFu), 0x5c00u);
    }

    // =================================================================
    // R4 - the .vf scalar delivered to Ara  (BUG 2, the vggnet16 blocker)
    //
    // OBSERVED: a .vf / .fs scalar reaches Ara with rs1[31:16] != 0xFFFF, so
    // ara_dispatcher.sv:2843-2844 substitutes 0x7e00. Hits bn1 channel 0
    // stably; moves between builds for T6c/T11/T17a/T17b.
    //
    // STATIC STATUS after reading the whole path on 2026-09-01: the decode is
    // correct (cva6_accel_first_pass_decoder.sv OPFVF -> is_fs1 -> ACCEL_OP_FS1),
    // ACCEL_OP_FS1/ACCEL_OP_FD are both correctly classified by is_rs1_fpr /
    // is_rd_fpr (ariane_pkg.sv:514,560) and sit outside every enum range used
    // by those functions, operand_a is correctly muxed to the FP regfile
    // (issue_read_operands.sv:1099), and Ara's own vfmv.f.s result IS boxed
    // (ara_dispatcher.sv:2238). So the static path is clean and the fault is
    // in what actually arrives, not in how it was meant to be assembled.
    //
    // R4 measures that directly: fmv.x.w reads the SAME f-register's 32-bit
    // pattern in the SAME block, immediately before the .vf consumes it.
    //   box=0xFFFF4200 but result=0x7e00 -> the f-register READ PATH is fine at
    //     that point, but Ara still saw an unboxed operand.
    //   box!=0xFFFF4200 -> the f-register itself is already wrong.
    // ★ CAVEAT (2026-09-01): this probe is NOT decisive against the commit-port
    // bug (see R5). `fmv.x.w` can be satisfied by WRITEBACK FORWARDING and so
    // return the correct value even when the architectural FPR is later written
    // as zero through commit port 1; the `.vf` that reads the regfile afterwards
    // would then still get zero. So a correct box here does NOT clear port 1.
    // R4 only distinguishes "read path clean at this instant" from "already
    // wrong"; R5 is the test that actually decides.
    // Both the probed and unprobed forms are run, because inserting the probe
    // changes the schedule and may itself hide the fault -- if it does, that
    // is also a result worth having.
    // =================================================================
    printf_("\n[R4] .vf scalar delivered to Ara (bug 2)\n");
    {
        static _Float16 out[8];
        unsigned box = 0, r0;
        long vl;
        asm volatile("vsetvli %0, %1, e16, m1, ta, ma" : "=r"(vl) : "r"(8L));

        // 3.0h = 0x4200. vfmv.v.f v8, 2.0h ; vfmv.v.f v16, 1.0h ;
        // vfmacc.vf v16, f(3.0), v8  ->  1 + 3*2 = 7.0h = 0x4700
        // R4a: unprobed, the T17a shape.
        asm volatile("li t1, 0x4000\n\tfmv.h.x ft0, t1\n\t"      // 2.0
                     "li t1, 0x3c00\n\tfmv.h.x ft1, t1\n\t"      // 1.0
                     "li t1, 0x4200\n\tfmv.h.x ft2, t1\n\t"      // 3.0
                     "vfmv.v.f v8,  ft0\n\t"
                     "vfmv.v.f v16, ft1\n\t"
                     "vfmacc.vf v16, ft2, v8\n\t"
                     "vse16.v v16, (%0)"
                     :: "r"(out) : "t1","ft0","ft1","ft2","memory");
        r0 = (unsigned)*(volatile unsigned short *)&out[0];
        chk16("R4a vfmacc.vf unprobed", r0, 0x4700u);

        // R4b: identical, but the scalar register is read back as raw bits
        // immediately before the vfmacc.vf that consumes it.
        asm volatile("li t1, 0x4000\n\tfmv.h.x ft0, t1\n\t"
                     "li t1, 0x3c00\n\tfmv.h.x ft1, t1\n\t"
                     "li t1, 0x4200\n\tfmv.h.x ft2, t1\n\t"
                     "vfmv.v.f v8,  ft0\n\t"
                     "vfmv.v.f v16, ft1\n\t"
                     "fmv.x.w %0, ft2\n\t"
                     "vfmacc.vf v16, ft2, v8\n\t"
                     "vse16.v v16, (%1)"
                     : "=r"(box) : "r"(out) : "t1","ft0","ft1","ft2","memory");
        r0 = (unsigned)*(volatile unsigned short *)&out[0];
        printf_("    f-reg box read at point of use = 0x%08x (want 0xffff4200)\n", box);
        chk16("R4b vfmacc.vf probed", r0, 0x4700u);

        // R4c: bn1's seed shape -- fmv.h.x -> vfmv.s.f -> vfredusum.vs.
        // v8 is filled with 1.0h at vl=8, so the reduction is 0 + 8 = 8.0h.
        static _Float16 red[8];
        unsigned rr, sbox = 0;
        asm volatile("li t1, 0x3c00\n\tfmv.h.x ft1, t1\n\t"
                     "vfmv.v.f v8, ft1\n\t"
                     "fmv.h.x ft3, zero\n\t"
                     "vfmv.s.f v16, ft3\n\t"
                     "vfredusum.vs v16, v8, v16\n\t"
                     "vse16.v v16, (%0)"
                     :: "r"(red) : "t1","ft1","ft3","memory");
        rr = (unsigned)*(volatile unsigned short *)&red[0];
        chk16("R4c bn1 seed shape", rr, 0x4800u);   // 8.0

        // R4d: same, probing the seed register at the point of use.
        asm volatile("li t1, 0x3c00\n\tfmv.h.x ft1, t1\n\t"
                     "vfmv.v.f v8, ft1\n\t"
                     "fmv.h.x ft3, zero\n\t"
                     "fmv.x.w %0, ft3\n\t"
                     "vfmv.s.f v16, ft3\n\t"
                     "vfredusum.vs v16, v8, v16\n\t"
                     "vse16.v v16, (%1)"
                     : "=r"(sbox) : "r"(red) : "t1","ft1","ft3","memory");
        rr = (unsigned)*(volatile unsigned short *)&red[0];
        printf_("    seed box at point of use = 0x%08x (want 0xffff0000)\n", sbox);
        chk16("R4d bn1 seed probed", rr, 0x4800u);
    }

    // =================================================================
    // R5 - FPR COMMIT-PORT CENSUS.  Direct probe for the bug found on
    // 2026-09-01 in issue_read_operands.sv:
    //
    //   :1029  logic [NrCommitPorts-1:0][CVA6Cfg.XLEN-1:0] fp_wdata_pack;
    //   :1048  assign fp_wdata_pack[i] = {wdata_i[i][CVA6Cfg.FLen-1:0]};
    //   :1063  .wdata_i (fp_wdata_pack)      -> ariane_regfile_fpga
    //   regfile port is [NrCommitPorts-1:0][DATA_WIDTH-1:0], DATA_WIDTH = FLen
    //
    // With RVD=0 the config is FLen=32 but XLEN=64, so fp_wdata_pack is
    // 2x64 = 128 bits while the regfile's wdata_i is 2x32 = 64 bits.
    // The 128->64 connection keeps the LOW 64 bits = fp_wdata_pack[0], so
    //   regfile wdata_i[0] = wdata_i[0][31:0]        (correct)
    //   regfile wdata_i[1] = fp_wdata_pack[0][63:32] (the zero-extension) = 0
    // => any FP result that retires on COMMIT PORT 1 writes ZERO.
    // waddr_pack and we_pack are both correctly sized, so the right register
    // is written -- with zero. Silent corruption.
    //
    // NrCommitPorts=2 and SuperscalarEn=0 in the compiled config
    // (cv64a6_imafdcv_sv39_config_pkg.sv:15,95), and CVA6 still retires up to
    // two instructions per cycle, so a run of back-to-back FP writes should
    // land roughly half of its results on port 1.
    //
    // This block writes the SAME constant into 8 f-registers back to back and
    // reads all 8 straight back. Any register reading 0x00000000 was written
    // through port 1. The nop padding shifts the commit alignment, so the
    // pattern of zeros should MOVE with the padding.
    // =================================================================
    printf_("\n[R5] FPR commit-port census\n");
    {
        static volatile unsigned res[8];
        static volatile unsigned src[8];
        const unsigned ONE  = 0x3f800000u;   // 1.0f
        const unsigned FOUR = 0x40800000u;   // 4.0f
        int bad = 0;
        for (int i = 0; i < 8; i++) src[i] = ONE;

        // Port 1 can only retire when port 0 retires in the SAME cycle
        // (commit_stage.sv:314 `commit_ack_o[0] && commit_instr_i[1].valid`),
        // and port 1's FU must be in {ALU, LOAD, CTRL_FLOW, MULT, FPU, FPU_VEC}
        // (:322). Both `fmv.w.x` (FPU) and `flw` (LOAD, is_rd_fpr) qualify.
        //
        // ★ So a run of back-to-back single-cycle FP moves is a POOR probe:
        // under single issue they complete one per cycle and mostly retire one
        // per cycle on port 0. The reliable shape is a LONG operation sitting
        // at the head of the scoreboard while YOUNGER independent FP results
        // complete behind it -- when the long one finally commits, a younger FP
        // result retires alongside it on port 1. That is exactly T7a's shape
        // (`fsqrt` + an independent numerator producer).
        //
        // Every readback is separated from its producer by 64 nops so the
        // scoreboard has drained: `fmv.x.w` then reads the ARCHITECTURAL
        // register file, not a writeback-forwarding path. Without that drain a
        // correct value would prove nothing (this is why R4 is not decisive).

        // R5a: naive back-to-back moves. A clean result here proves NOTHING --
        // these may all retire through port 0. Kept only as a contrast to R5b.
        asm volatile(
            "fmv.w.x ft1, %1\n\tfmv.w.x ft2, %1\n\t"
            "fmv.w.x ft3, %1\n\tfmv.w.x ft4, %1\n\t"
            ".rept 64\n\tnop\n\t.endr\n\t"
            "fmv.x.w t2, ft1\n\tsw t2,  0(%0)\n\t"
            "fmv.x.w t2, ft2\n\tsw t2,  4(%0)\n\t"
            "fmv.x.w t2, ft3\n\tsw t2,  8(%0)\n\t"
            "fmv.x.w t2, ft4\n\tsw t2, 12(%0)"
            :: "r"(res), "r"(ONE)
            : "t2","ft1","ft2","ft3","ft4","memory");
        printf_("    R5a back-to-back moves :");
        for (int k = 0; k < 4; k++)
            printf_(" %s", res[k] == ONE ? "ok" : (res[k] == 0 ? "ZERO" : "??"));
        printf_("   (a clean row here proves nothing)\n");

        // R5b: ★ the real probe. A long fdiv.s holds the head while four
        // independent fmv.w.x complete behind it.
        asm volatile(
            "fmv.w.x ft8, %1\n\t"
            "fmv.w.x ft9, %2\n\t"
            "fdiv.s  ft10, ft8, ft9\n\t"          /* LONG: 1.0/4.0 */
            "fmv.w.x ft1, %1\n\tfmv.w.x ft2, %1\n\t"
            "fmv.w.x ft3, %1\n\tfmv.w.x ft4, %1\n\t"
            ".rept 64\n\tnop\n\t.endr\n\t"
            "fmv.x.w t2, ft1\n\tsw t2,  0(%0)\n\t"
            "fmv.x.w t2, ft2\n\tsw t2,  4(%0)\n\t"
            "fmv.x.w t2, ft3\n\tsw t2,  8(%0)\n\t"
            "fmv.x.w t2, ft4\n\tsw t2, 12(%0)\n\t"
            "fmv.x.w t2, ft10\n\tsw t2, 16(%0)"
            :: "r"(res), "r"(ONE), "r"(FOUR)
            : "t2","ft1","ft2","ft3","ft4","ft8","ft9","ft10","memory");
        printf_("    R5b fdiv + younger FP  :");
        for (int k = 0; k < 4; k++) {
            printf_(" %s", res[k] == ONE ? "ok" : (res[k] == 0 ? "ZERO" : "??"));
            if (res[k] != ONE) bad++;
        }
        printf_("   quotient=0x%08x (want 0x3e800000)\n", res[4]);
        if (res[4] != 0x3e800000u) bad++;

        // R5c: same shape but the younger producers are LOADS -- T7a's actual
        // object code is `fsqrt.s` + an independent `flw` numerator.
        asm volatile(
            "fmv.w.x ft8, %1\n\t"
            "fmv.w.x ft9, %2\n\t"
            "fdiv.s  ft10, ft8, ft9\n\t"          /* LONG */
            "flw ft1, 0(%3)\n\tflw ft2, 4(%3)\n\t"
            "flw ft3, 8(%3)\n\tflw ft4, 12(%3)\n\t"
            ".rept 64\n\tnop\n\t.endr\n\t"
            "fmv.x.w t2, ft1\n\tsw t2,  0(%0)\n\t"
            "fmv.x.w t2, ft2\n\tsw t2,  4(%0)\n\t"
            "fmv.x.w t2, ft3\n\tsw t2,  8(%0)\n\t"
            "fmv.x.w t2, ft4\n\tsw t2, 12(%0)"
            :: "r"(res), "r"(ONE), "r"(FOUR), "r"(src)
            : "t2","ft1","ft2","ft3","ft4","ft8","ft9","ft10","memory");
        printf_("    R5c fdiv + younger flw :");
        for (int k = 0; k < 4; k++) {
            printf_(" %s", res[k] == ONE ? "ok" : (res[k] == 0 ? "ZERO" : "??"));
            if (res[k] != ONE) bad++;
        }
        printf_("\n");

        // R5d: fsqrt instead of fdiv -- T7a's exact long operation.
        asm volatile(
            "fmv.w.x ft9, %2\n\t"
            "fsqrt.s ft10, ft9\n\t"               /* LONG: sqrt(4.0) */
            "fmv.w.x ft1, %1\n\tfmv.w.x ft2, %1\n\t"
            "fmv.w.x ft3, %1\n\tfmv.w.x ft4, %1\n\t"
            ".rept 64\n\tnop\n\t.endr\n\t"
            "fmv.x.w t2, ft1\n\tsw t2,  0(%0)\n\t"
            "fmv.x.w t2, ft2\n\tsw t2,  4(%0)\n\t"
            "fmv.x.w t2, ft3\n\tsw t2,  8(%0)\n\t"
            "fmv.x.w t2, ft4\n\tsw t2, 12(%0)"
            :: "r"(res), "r"(ONE), "r"(FOUR)
            : "t2","ft1","ft2","ft3","ft4","ft9","ft10","memory");
        printf_("    R5d fsqrt + younger FP :");
        for (int k = 0; k < 4; k++) {
            printf_(" %s", res[k] == ONE ? "ok" : (res[k] == 0 ? "ZERO" : "??"));
            if (res[k] != ONE) bad++;
        }
        printf_("\n");

        chk32("R5 FPR writes intact (b,c,d)", (unsigned)bad, 0u);
    }

    printf_("\n===== fpubug done: %d failure(s) =====\n", fails);
    return 0;
}
