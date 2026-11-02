#include <stdint.h>
#include <string.h>

#ifdef SPIKE
#include <printf.h>
#elif defined ARA_LINUX
#include <stdio.h>
#define printf_ printf
#else
#include "printf.h"
#endif

#define ITER        10
#define ARRAY_BYTES 128
#define FP16_ELEMS  (ARRAY_BYTES / 2)   /* 64  — fills one m1 register at e16 */
#define FP32_ELEMS  (ARRAY_BYTES / 4)   /* 32  — fills one m1 register at e32 */
#define FP64_ELEMS  (ARRAY_BYTES / 8)   /* 16  — fills one m1 register at e64 */

/* ---------- fp16 section arrays — registers v8, v12, v16, v24 ---------- */
uint16_t __attribute__((aligned(64))) data_a[FP16_ELEMS];      /* fp16 1.0 = 0x3C00      */
uint16_t __attribute__((aligned(64))) data_b[FP16_ELEMS];      /* fp16 2.0 = 0x4000      */
uint32_t __attribute__((aligned(64))) data_acc[FP32_ELEMS];    /* fp32 1.0 = 0x3F800000  */
uint16_t __attribute__((aligned(64))) gather_idx[FP16_ELEMS];  /* byte offsets, e16      */

/* ---------- fp32 section arrays — registers v0, v4, v20, v28 ----------- */
uint32_t __attribute__((aligned(64))) data_a32[FP32_ELEMS];    /* fp32 1.0 = 0x3F800000         */
uint32_t __attribute__((aligned(64))) data_b32[FP32_ELEMS];    /* fp32 2.0 = 0x40000000         */
uint64_t __attribute__((aligned(64))) data_acc64[FP64_ELEMS];  /* fp64 1.0 = 0x3FF0000000000000 */
uint32_t __attribute__((aligned(64))) gather_idx32[FP32_ELEMS];/* byte offsets, e32             */

static inline int64_t fc_cycle_count_local(void)
{
    int64_t cycle_count = 0;
    asm volatile("fence; csrr %0, cycle" : "=r"(cycle_count));
    return cycle_count;
}

int main(void)
{
    int64_t t0, t1, diff;
    size_t vl;
    float dummy_sync;

    for (int i = 0; i < FP16_ELEMS; i++) {
        data_a[i]      = 0x3C00;
        data_b[i]      = 0x4000;
        gather_idx[i]  = (uint16_t)(((i * 2) % FP16_ELEMS) * sizeof(uint16_t));
    }
    for (int i = 0; i < FP32_ELEMS; i++) {
        data_acc[i]      = 0x3F800000;
        data_a32[i]      = 0x3F800000;
        data_b32[i]      = 0x40000000;
        gather_idx32[i]  = (uint32_t)(((i * 2) % FP32_ELEMS) * sizeof(uint32_t));
    }
    for (int i = 0; i < FP64_ELEMS; i++)
        data_acc64[i] = 0x3FF0000000000000ULL;

    /* ================================================================
     *  FP16 SECTION  (VLEN=128B, LMUL=1 → 1 register = 128 bytes)
     *  Registers: v8, v12, v16, v24
     * ================================================================ */
    asm volatile("vsetvli %0, %1, e16, m1, ta, ma" : "=r"(vl) : "r"((size_t)FP16_ELEMS));

    printf_("==========================================\n");
    printf_("   ARA FP16 VECTOR COHERENCE TEST         \n");
    printf_("==========================================\n");
    printf_("VLEN=128B, LMUL=1 | Array: %d bytes\n", ARRAY_BYTES);
    printf_("VL (e16, m1): %d  |  ITER: %d\n\n", (int)vl, ITER);

    /* warmup: v8=fp16, v16=fp16, v24=gather indices */
    asm volatile("vle16.v v8,  (%0)" :: "r"(data_a));
    asm volatile("vle16.v v16, (%0)" :: "r"(data_b));
    asm volatile("vle16.v v24, (%0)" :: "r"(gather_idx));

    /* ----------------------------------------------------------
     * 1. UNIT-STRIDE LOAD (vle16.v)   VL=64
     * ---------------------------------------------------------- */
    t0 = fc_cycle_count_local();
    for (int i = 0; i < ITER; i++)
        asm volatile("vle16.v v8, (%0)" :: "r"(data_a));
    asm volatile("fence");
    t1 = fc_cycle_count_local();
    diff = t1 - t0;
    printf_("[vle16.v]    Unit-Stride Load : %6ld cyc | ~%4.1f cyc/inst\n",
            (long)diff, (float)diff / ITER);

    /* ----------------------------------------------------------
     * 2. STRIDED LOAD (vlse16.v)   VL=64
     * ---------------------------------------------------------- */
    long stride = 4;
    t0 = fc_cycle_count_local();
    for (int i = 0; i < ITER; i++)
        asm volatile("vlse16.v v8, (%0), %1" :: "r"(data_a), "r"(stride));
    asm volatile("fence");
    t1 = fc_cycle_count_local();
    diff = t1 - t0;
    printf_("[vlse16.v]   Strided Load     : %6ld cyc | ~%4.1f cyc/inst\n",
            (long)diff, (float)diff / ITER);

    /* ----------------------------------------------------------
     * 3. GATHER LOAD (vluxei16.v)   VL=64
     * ---------------------------------------------------------- */
    t0 = fc_cycle_count_local();
    for (int i = 0; i < ITER; i++)
        asm volatile("vluxei16.v v8, (%0), v24" :: "r"(data_a));
    asm volatile("fence");
    t1 = fc_cycle_count_local();
    diff = t1 - t0;
    printf_("[vluxei16.v] Gather Load      : %6ld cyc | ~%4.1f cyc/inst\n",
            (long)diff, (float)diff / ITER);

    /* ----------------------------------------------------------
     * 4. VECTOR ADD (vfadd.vv)   VL=64
     * ---------------------------------------------------------- */
    asm volatile("vle16.v v8,  (%0)" :: "r"(data_a));
    asm volatile("vle16.v v16, (%0)" :: "r"(data_b));
    t0 = fc_cycle_count_local();
    for (int i = 0; i < ITER; i++)
        asm volatile("vfadd.vv v8, v8, v16");  /* v8 = v8 + v16 (fp16) */
    asm volatile("vfmv.f.s %0, v8" : "=f"(dummy_sync));
    t1 = fc_cycle_count_local();
    diff = t1 - t0;
    printf_("[vfadd.vv   e16] fp16+fp16->fp16       : %6ld cyc | ~%4.1f cyc/inst\n",
            (long)diff, (float)diff / ITER);

    /* ----------------------------------------------------------
     * 5. VECTOR MUL (vfmul.vv)   VL=64
     * ---------------------------------------------------------- */
    asm volatile("vle16.v v8, (%0)" :: "r"(data_a));
    t0 = fc_cycle_count_local();
    for (int i = 0; i < ITER; i++)
        asm volatile("vfmul.vv v8, v8, v16");  /* v8 = v8 * v16 (fp16) */
    asm volatile("vfmv.f.s %0, v8" : "=f"(dummy_sync));
    t1 = fc_cycle_count_local();
    diff = t1 - t0;
    printf_("[vfmul.vv   e16] fp16*fp16->fp16       : %6ld cyc | ~%4.1f cyc/inst\n",
            (long)diff, (float)diff / ITER);

    /* ----------------------------------------------------------
     * 6. vfncvt.f.f.w — TAIL AGNOSTIC (ta)   VL=32
     *
     *    6a: same register — vd=v8(e16,m1) is the lower half of
     *        vs2=v8(e32,m2); valid per RVV spec for narrowing ops.
     *    6b: v16 was loaded as fp16, never set up as fp32;
     *        hardware reinterprets the fp16 bit patterns as fp32.
     * ---------------------------------------------------------- */
    printf_("\n--- vfncvt.f.f.w [Tail Agnostic] ---\n");

    /* load fp32 source into v8 with e32,m1 (VL=32, 128 bytes = 1 register) */
    asm volatile("vsetvli %0, %1, e32, m1, ta, ma" : "=r"(vl) : "r"((size_t)FP32_ELEMS));
    asm volatile("vle32.v v8, (%0)" :: "r"(data_acc));

    /* switch to e16,m1 for narrowing; vs2 is implicitly e32,m2 = {v8,v9} */
    asm volatile("vsetvli %0, %1, e16, m1, ta, ma" : "=r"(vl) : "r"((size_t)FP32_ELEMS));
    printf_("  VL (e16, m1, ta): %d\n", (int)vl);

    t0 = fc_cycle_count_local();
    for (int i = 0; i < ITER; i++)
        asm volatile("vfncvt.f.f.w v8, v8");   /* fp32(v8,m2) -> fp16(v8,m1) */
    asm volatile("vfmv.f.s %0, v8" : "=f"(dummy_sync));
    t1 = fc_cycle_count_local();
    diff = t1 - t0;
    printf_("  [ta] same-reg   v8(fp32,m2)   -> v8(fp16,m1)  : %6ld cyc | ~%4.1f cyc/inst\n",
            (long)diff, (float)diff / ITER);

    t0 = fc_cycle_count_local();
    for (int i = 0; i < ITER; i++)
        asm volatile("vfncvt.f.f.w v8, v16");  /* v16 holds fp16 data, reinterpreted as fp32 */
    asm volatile("vfmv.f.s %0, v8" : "=f"(dummy_sync));
    t1 = fc_cycle_count_local();
    diff = t1 - t0;
    printf_("  [ta] uninit-src v16(!fp32,m2) -> v8(fp16,m1)  : %6ld cyc | ~%4.1f cyc/inst\n",
            (long)diff, (float)diff / ITER);

    /* ----------------------------------------------------------
     * 5. vfncvt.f.f.w — TAIL UNDISTURBED (tu)   VL=32
     * ---------------------------------------------------------- */
    printf_("\n--- vfncvt.f.f.w [Tail Undisturbed] ---\n");

    asm volatile("vsetvli %0, %1, e32, m1, tu, ma" : "=r"(vl) : "r"((size_t)FP32_ELEMS));
    asm volatile("vle32.v v8, (%0)" :: "r"(data_acc));

    asm volatile("vsetvli %0, %1, e16, m1, tu, ma" : "=r"(vl) : "r"((size_t)FP32_ELEMS));
    printf_("  VL (e16, m1, tu): %d\n", (int)vl);

    t0 = fc_cycle_count_local();
    for (int i = 0; i < ITER; i++)
        asm volatile("vfncvt.f.f.w v8, v8");
    asm volatile("vfmv.f.s %0, v8" : "=f"(dummy_sync));
    t1 = fc_cycle_count_local();
    diff = t1 - t0;
    printf_("  [tu] same-reg   v8(fp32,m2)   -> v8(fp16,m1)  : %6ld cyc | ~%4.1f cyc/inst\n",
            (long)diff, (float)diff / ITER);

    t0 = fc_cycle_count_local();
    for (int i = 0; i < ITER; i++)
        asm volatile("vfncvt.f.f.w v8, v16");
    asm volatile("vfmv.f.s %0, v8" : "=f"(dummy_sync));
    t1 = fc_cycle_count_local();
    diff = t1 - t0;
    printf_("  [tu] uninit-src v16(!fp32,m2) -> v8(fp16,m1)  : %6ld cyc | ~%4.1f cyc/inst\n",
            (long)diff, (float)diff / ITER);

    /* ----------------------------------------------------------
     * 6. FP16 -> FP16 MACC (vfmacc.vv, e16)   VL=64
     *    vd(fp16) += vs1(fp16) * vs2(fp16)
     * ---------------------------------------------------------- */
    printf_("\n--- FP16 MACC ---\n");

    asm volatile("vsetvli %0, %1, e16, m1, ta, ma" : "=r"(vl) : "r"((size_t)FP16_ELEMS));
    printf_("  VL (e16, m1): %d\n", (int)vl);

    asm volatile("vle16.v v8,  (%0)" :: "r"(data_a));  /* fp16 1.0 */
    asm volatile("vle16.v v16, (%0)" :: "r"(data_b));  /* fp16 2.0 */

    t0 = fc_cycle_count_local();
    for (int i = 0; i < ITER; i++)
        asm volatile("vfmacc.vv v8, v16, v8");  /* v8 += v16 * v8 (fp16) */
    asm volatile("vfmv.f.s %0, v8" : "=f"(dummy_sync));
    t1 = fc_cycle_count_local();
    diff = t1 - t0;
    printf_("  [vfmacc.vv  e16] fp16*fp16+fp16->fp16 : %6ld cyc | ~%4.1f cyc/inst\n",
            (long)diff, (float)diff / ITER);

    /* ----------------------------------------------------------
     * 7. WIDENING MACC (vfwmacc.vv, e16->e32)   VL=32
     *    vd(fp32,m2) += vs1(fp16,m1) * vs2(fp16,m1)
     *    Active vtype e16,m1: vd is implicitly m2 = {v16,v17}.
     *    With VL=32, only v16 (first 128B) is accessed — one register.
     *    Register groups: v8(m1), v12(m1), v16(m2) — no overlaps.
     * ---------------------------------------------------------- */
    printf_("\n--- Widening FP16*FP16 -> FP32 MACC ---\n");

    asm volatile("vsetvli %0, %1, e16, m1, ta, ma" : "=r"(vl) : "r"((size_t)FP32_ELEMS));
    printf_("  VL (e16, m1, widens to fp32): %d\n", (int)vl);

    asm volatile("vle16.v v8,  (%0)" :: "r"(data_a));  /* v8(fp16,m1)  */
    asm volatile("vle16.v v12, (%0)" :: "r"(data_b));  /* v12(fp16,m1) */

    /* load fp32 accumulator: e32,m1 fills v16 exactly (32 elements = 128B) */
    asm volatile("vsetvli zero, %0, e32, m1, ta, ma" :: "r"((size_t)FP32_ELEMS));
    asm volatile("vle32.v v16, (%0)" :: "r"(data_acc));

    asm volatile("vsetvli zero, %0, e16, m1, ta, ma" :: "r"((size_t)FP32_ELEMS));

    t0 = fc_cycle_count_local();
    for (int i = 0; i < ITER; i++)
        asm volatile("vfwmacc.vv v16, v8, v12");  /* v16(fp32,m2) += v8(fp16) * v12(fp16) */
    asm volatile("vsetvli zero, %0, e32, m1, ta, ma" :: "r"((size_t)FP32_ELEMS));
    asm volatile("vfmv.f.s %0, v16" : "=f"(dummy_sync));
    t1 = fc_cycle_count_local();
    diff = t1 - t0;
    printf_("  [vfwmacc.vv e16] fp16*fp16+fp32->fp32 : %6ld cyc | ~%4.1f cyc/inst\n",
            (long)diff, (float)diff / ITER);

    /* ================================================================
     *  FP32 SECTION  (VLEN=128B, LMUL=1 → 1 register = 128 bytes)
     *  Registers: v0, v4, v20, v28
     * ================================================================ */
    asm volatile("vsetvli %0, %1, e32, m1, ta, ma" : "=r"(vl) : "r"((size_t)FP32_ELEMS));

    printf_("\n==========================================\n");
    printf_("   ARA FP32 VECTOR COHERENCE TEST         \n");
    printf_("==========================================\n");
    printf_("VLEN=128B, LMUL=1 | Array: %d bytes\n", ARRAY_BYTES);
    printf_("VL (e32, m1): %d  |  ITER: %d\n\n", (int)vl, ITER);

    /* warmup: v0=fp32, v4=fp32 (never set up as fp64), v28=gather indices */
    asm volatile("vle32.v v0,  (%0)" :: "r"(data_a32));
    asm volatile("vle32.v v4,  (%0)" :: "r"(data_b32));
    asm volatile("vle32.v v28, (%0)" :: "r"(gather_idx32));

    /* ----------------------------------------------------------
     * 8. UNIT-STRIDE LOAD (vle32.v)   VL=32
     * ---------------------------------------------------------- */
    t0 = fc_cycle_count_local();
    for (int i = 0; i < ITER; i++)
        asm volatile("vle32.v v0, (%0)" :: "r"(data_a32));
    asm volatile("fence");
    t1 = fc_cycle_count_local();
    diff = t1 - t0;
    printf_("[vle32.v]    Unit-Stride Load : %6ld cyc | ~%4.1f cyc/inst\n",
            (long)diff, (float)diff / ITER);

    /* ----------------------------------------------------------
     * 9. STRIDED LOAD (vlse32.v)   VL=32
     * ---------------------------------------------------------- */
    long stride32 = 8;
    t0 = fc_cycle_count_local();
    for (int i = 0; i < ITER; i++)
        asm volatile("vlse32.v v0, (%0), %1" :: "r"(data_a32), "r"(stride32));
    asm volatile("fence");
    t1 = fc_cycle_count_local();
    diff = t1 - t0;
    printf_("[vlse32.v]   Strided Load     : %6ld cyc | ~%4.1f cyc/inst\n",
            (long)diff, (float)diff / ITER);

    /* ----------------------------------------------------------
     * 10. GATHER LOAD (vluxei32.v)   VL=32
     * ---------------------------------------------------------- */
    t0 = fc_cycle_count_local();
    for (int i = 0; i < ITER; i++)
        asm volatile("vluxei32.v v0, (%0), v28" :: "r"(data_a32));
    asm volatile("fence");
    t1 = fc_cycle_count_local();
    diff = t1 - t0;
    printf_("[vluxei32.v] Gather Load      : %6ld cyc | ~%4.1f cyc/inst\n",
            (long)diff, (float)diff / ITER);

    /* ----------------------------------------------------------
     * 11. VECTOR ADD (vfadd.vv)   VL=32
     * ---------------------------------------------------------- */
    asm volatile("vle32.v v0, (%0)" :: "r"(data_a32));
    asm volatile("vle32.v v4, (%0)" :: "r"(data_b32));
    t0 = fc_cycle_count_local();
    for (int i = 0; i < ITER; i++)
        asm volatile("vfadd.vv v0, v0, v4");  /* v0 = v0 + v4 (fp32) */
    asm volatile("vfmv.f.s %0, v0" : "=f"(dummy_sync));
    t1 = fc_cycle_count_local();
    diff = t1 - t0;
    printf_("[vfadd.vv   e32] fp32+fp32->fp32       : %6ld cyc | ~%4.1f cyc/inst\n",
            (long)diff, (float)diff / ITER);

    /* ----------------------------------------------------------
     * 12. VECTOR MUL (vfmul.vv)   VL=32
     * ---------------------------------------------------------- */
    asm volatile("vle32.v v0, (%0)" :: "r"(data_a32));
    t0 = fc_cycle_count_local();
    for (int i = 0; i < ITER; i++)
        asm volatile("vfmul.vv v0, v0, v4");  /* v0 = v0 * v4 (fp32) */
    asm volatile("vfmv.f.s %0, v0" : "=f"(dummy_sync));
    t1 = fc_cycle_count_local();
    diff = t1 - t0;
    printf_("[vfmul.vv   e32] fp32*fp32->fp32       : %6ld cyc | ~%4.1f cyc/inst\n",
            (long)diff, (float)diff / ITER);

    /* ----------------------------------------------------------
     * 13. vfncvt.f.f.w — TAIL AGNOSTIC (ta)   VL=16
     *
     *    13a: same register — vd=v0(e32,m1) is the lower half of
     *         vs2=v0(e64,m2); valid per RVV spec for narrowing ops.
     *    13b: v4 was loaded as fp32, never set up as fp64;
     *         hardware reinterprets the fp32 bit patterns as fp64.
     * ---------------------------------------------------------- */
    printf_("\n--- vfncvt.f.f.w [Tail Agnostic] ---\n");

    /* load fp64 source into v0 with e64,m1 (VL=16, 128 bytes = 1 register) */
    asm volatile("vsetvli %0, %1, e64, m1, ta, ma" : "=r"(vl) : "r"((size_t)FP64_ELEMS));
    asm volatile("vle64.v v0, (%0)" :: "r"(data_acc64));

    /* switch to e32,m1; vs2 is implicitly e64,m2 = {v0,v1} */
    asm volatile("vsetvli %0, %1, e32, m1, ta, ma" : "=r"(vl) : "r"((size_t)FP64_ELEMS));
    printf_("  VL (e32, m1, ta): %d\n", (int)vl);

    t0 = fc_cycle_count_local();
    for (int i = 0; i < ITER; i++)
        asm volatile("vfncvt.f.f.w v0, v0");   /* fp64(v0,m2) -> fp32(v0,m1) */
    asm volatile("vfmv.f.s %0, v0" : "=f"(dummy_sync));
    t1 = fc_cycle_count_local();
    diff = t1 - t0;
    printf_("  [ta] same-reg   v0(fp64,m2)   -> v0(fp32,m1)  : %6ld cyc | ~%4.1f cyc/inst\n",
            (long)diff, (float)diff / ITER);

    t0 = fc_cycle_count_local();
    for (int i = 0; i < ITER; i++)
        asm volatile("vfncvt.f.f.w v0, v4");   /* v4 holds fp32 data, reinterpreted as fp64 */
    asm volatile("vfmv.f.s %0, v0" : "=f"(dummy_sync));
    t1 = fc_cycle_count_local();
    diff = t1 - t0;
    printf_("  [ta] uninit-src v4(!fp64,m2)  -> v0(fp32,m1)  : %6ld cyc | ~%4.1f cyc/inst\n",
            (long)diff, (float)diff / ITER);

    /* ----------------------------------------------------------
     * 12. vfncvt.f.f.w — TAIL UNDISTURBED (tu)   VL=16
     * ---------------------------------------------------------- */
    printf_("\n--- vfncvt.f.f.w [Tail Undisturbed] ---\n");

    asm volatile("vsetvli %0, %1, e64, m1, tu, ma" : "=r"(vl) : "r"((size_t)FP64_ELEMS));
    asm volatile("vle64.v v0, (%0)" :: "r"(data_acc64));

    asm volatile("vsetvli %0, %1, e32, m1, tu, ma" : "=r"(vl) : "r"((size_t)FP64_ELEMS));
    printf_("  VL (e32, m1, tu): %d\n", (int)vl);

    t0 = fc_cycle_count_local();
    for (int i = 0; i < ITER; i++)
        asm volatile("vfncvt.f.f.w v0, v0");
    asm volatile("vfmv.f.s %0, v0" : "=f"(dummy_sync));
    t1 = fc_cycle_count_local();
    diff = t1 - t0;
    printf_("  [tu] same-reg   v0(fp64,m2)   -> v0(fp32,m1)  : %6ld cyc | ~%4.1f cyc/inst\n",
            (long)diff, (float)diff / ITER);

    t0 = fc_cycle_count_local();
    for (int i = 0; i < ITER; i++)
        asm volatile("vfncvt.f.f.w v0, v4");
    asm volatile("vfmv.f.s %0, v0" : "=f"(dummy_sync));
    t1 = fc_cycle_count_local();
    diff = t1 - t0;
    printf_("  [tu] uninit-src v4(!fp64,m2)  -> v0(fp32,m1)  : %6ld cyc | ~%4.1f cyc/inst\n",
            (long)diff, (float)diff / ITER);

    /* ----------------------------------------------------------
     * 13. FP32 -> FP32 MACC (vfmacc.vv, e32)   VL=32
     *     vd(fp32) += vs1(fp32) * vs2(fp32)
     * ---------------------------------------------------------- */
    printf_("\n--- FP32 MACC ---\n");

    asm volatile("vsetvli %0, %1, e32, m1, ta, ma" : "=r"(vl) : "r"((size_t)FP32_ELEMS));
    printf_("  VL (e32, m1): %d\n", (int)vl);

    asm volatile("vle32.v v0, (%0)" :: "r"(data_a32));  /* fp32 1.0 */
    asm volatile("vle32.v v4, (%0)" :: "r"(data_b32));  /* fp32 2.0 */

    t0 = fc_cycle_count_local();
    for (int i = 0; i < ITER; i++)
        asm volatile("vfmacc.vv v0, v4, v0");  /* v0 += v4 * v0 (fp32) */
    asm volatile("vfmv.f.s %0, v0" : "=f"(dummy_sync));
    t1 = fc_cycle_count_local();
    diff = t1 - t0;
    printf_("  [vfmacc.vv  e32] fp32*fp32+fp32->fp32 : %6ld cyc | ~%4.1f cyc/inst\n",
            (long)diff, (float)diff / ITER);

    /* ----------------------------------------------------------
     * 14. WIDENING MACC (vfwmacc.vv, e32->e64)   VL=16
     *     vd(fp64,m2) += vs1(fp32,m1) * vs2(fp32,m1)
     *     Active vtype e32,m1: vd is implicitly m2 = {v20,v21}.
     *     With VL=16, only v20 (first 128B) is accessed — one register.
     *     Register groups: v0(m1), v4(m1), v20(m2) — no overlaps.
     * ---------------------------------------------------------- */
    printf_("\n--- Widening FP32*FP32 -> FP64 MACC ---\n");

    asm volatile("vsetvli %0, %1, e32, m1, ta, ma" : "=r"(vl) : "r"((size_t)FP64_ELEMS));
    printf_("  VL (e32, m1, widens to fp64): %d\n", (int)vl);

    asm volatile("vle32.v v0, (%0)" :: "r"(data_a32));  /* v0(fp32,m1) */
    asm volatile("vle32.v v4, (%0)" :: "r"(data_b32));  /* v4(fp32,m1) */

    /* load fp64 accumulator: e64,m1 fills v20 exactly (16 elements = 128B) */
    asm volatile("vsetvli zero, %0, e64, m1, ta, ma" :: "r"((size_t)FP64_ELEMS));
    asm volatile("vle64.v v20, (%0)" :: "r"(data_acc64));

    asm volatile("vsetvli zero, %0, e32, m1, ta, ma" :: "r"((size_t)FP64_ELEMS));

    t0 = fc_cycle_count_local();
    for (int i = 0; i < ITER; i++)
        asm volatile("vfwmacc.vv v20, v0, v4");  /* v20(fp64,m2) += v0(fp32) * v4(fp32) */
    asm volatile("vsetvli zero, %0, e64, m1, ta, ma" :: "r"((size_t)FP64_ELEMS));
    asm volatile("vfmv.f.s %0, v20" : "=f"(dummy_sync));
    t1 = fc_cycle_count_local();
    diff = t1 - t0;
    printf_("  [vfwmacc.vv e32] fp32*fp32+fp64->fp64 : %6ld cyc | ~%4.1f cyc/inst\n",
            (long)diff, (float)diff / ITER);

    printf_("\n==========================================\n");
    return 0;
}
