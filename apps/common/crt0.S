// Copyright 2020 ETH Zurich and University of Bologna.
//
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Author: Samuel Riedel, ETH Zurich
//         Matheus Cavalcante, ETH Zurich

.globl _start
.globl _eoc
.globl _fail
.section .text;
.section .text.init;

#include "encoding.h"

// For the riscv-tests environment
.weak mtvec_handler
.weak stvec_handler
.weak rvtest_init

_start:
    // Initialize global pointer
    .option push
    .option norelax
    1:auipc gp, %pcrel_hi(__global_pointer$)
      addi  gp, gp, %pcrel_lo(1b)
    .option pop

    /* reset vector */
    j reset_vector
reset_vector:
#ifdef FPGA
    // ---- STALE-CODE GUARD (must be FIRST, and byte-identical in every app) --
    // The ILA proved (3 captures, 2026-07-28) that after a HUNG run the next
    // program can boot into the PREVIOUS program's code: the loader writes DDR
    // through the PS, invisible to the PL-side caches, and a cache that
    // survives the gpio412 reset (LLC tag init and/or CVA6 I$ valid bits on
    // FPGA) keeps serving the old lines. The shared crt0 prefix executes the
    // same either way -- the first program-SPECIFIC bytes are the `la gp` and
    // `la t0, main` immediates, so a stale line there vectors execution into
    // the OLD program's main (observed: vfcount run executing fmatmul@0x956).
    // Therefore: everything from here down to the post-flush refetch must stay
    // BYTE-IDENTICAL across programs, and main/gp must only be materialized
    // AFTER this block.
    fence.i                     // invalidate CVA6 I$: refetch all .text
// DIAGNOSTIC 2026-08-28: NO_BOOT_LLC_FLUSH skips ONLY this boot-time flush,
// keeping the _eoc flush (line ~258) so results still reach DDR. Added to test
// whether this poll hangs on the LLC-1MiB bitstream (2048 lines/way vs 256):
// fmatmul.fpga.bin, which contains this exact code, completes on the 08-21
// LLC-128KiB bitstream and never writes its sentinel on the 08-25 LLC-1MiB one.
// NOTE: this flush is also the STALE-CODE GUARD (see block comment above), so
// with it off, always reprogram the FPGA before a run to start with a clean LLC.
#if !defined(NO_LLC) && !defined(NO_BOOT_LLC_FLUSH)
    li   t0, 0x03001000         // axi_llc cfg (uncached reg bus: safe while
    lw   t4, 0x28(t0)           //   the LLC AXI port is isolated during flush)
    li   t1, 1
    sll  t1, t1, t4
    addi t1, t1, -1             // mask = (1<<ways)-1
    sw   t1, 0x8(t0)            // CFG_FLUSH_LOW  = mask
    sw   x0, 0xc(t0)            // CFG_FLUSH_HIGH = 0
    li   t2, 1
    sw   t2, 0x10(t0)           // COMMIT -> writeback+INVALIDATE every way
1:  lw   t3, 0x8(t0)            // poll CFG_FLUSH (hardware clears when done;
    bnez t3, 1b                 //   FLUSHED@0x18 is a live bitmap, NOT a flag)
#endif
    fence.i                     // refetch: from here on all code is FRESH
    // gp was set at _start possibly from a STALE line -- redo it now that
    // fetches are guaranteed fresh.
    .option push
    .option norelax
    2:auipc gp, %pcrel_hi(__global_pointer$)
      addi  gp, gp, %pcrel_lo(2b)
    .option pop
#endif
    li      x1, 0
    li      x4, 0
    li      x5, 0
    li      x6, 0
    li      x7, 0
    li      x8, 0
    li      x9, 0
    li      x10, 0
    li      x11, 0
    li      x12, 0
    li      x13, 0
    li      x14, 0
    li      x15, 0
    li      x16, 0
    li      x17, 0
    li      x18, 0
    li      x19, 0
    li      x20, 0
    li      x21, 0
    li      x22, 0
    li      x23, 0
    li      x24, 0
    li      x25, 0
    li      x26, 0
    li      x27, 0
    li      x28, 0
    li      x29, 0
    li      x30, 0
    li      x31, 0
    // Initialize stack at the end of the DRAM region
#ifdef FPGA
    // No TB to pre-load dram_end_address_reg; use _stack_top from linker script
    la      sp, _stack_top
#else
    la      t0, dram_end_address_reg
    ld      sp, 0(t0)
#endif
    // Set up a PMP to permit all accesses
    li t0, (1 << (31 + (__riscv_xlen / 64) * (53 - 31))) - 1
    csrw pmpaddr0, t0
    li t0, PMP_NAPOT | PMP_R | PMP_W | PMP_X
    csrw pmpcfg0, t0
    // Delegate no traps
    csrwi   mie, 0
    la      t0, trap_vector
    csrw    mtvec, t0
    csrwi   medeleg, 0
    csrwi   mideleg, 0
#ifndef FPGA
    // If an stvec_handler is defined, delegate exceptions to it
    // (skipped on FPGA: weak symbol resolves to 0x0, AUIPC PCREL from
    //  0xC0000000 would be -3 GB, out of the ±2 GB range)
    la t0, stvec_handler
    beqz t0, 1f
    csrw stvec, t0
    li t0, (1 << CAUSE_LOAD_PAGE_FAULT) |  \
           (1 << CAUSE_STORE_PAGE_FAULT) | \
           (1 << CAUSE_FETCH_PAGE_FAULT) | \
           (1 << CAUSE_MISALIGNED_FETCH) | \
           (1 << CAUSE_USER_ECALL) |       \
           (1 << CAUSE_BREAKPOINT)
    csrw medeleg, t0
#endif
    .align 2
#ifdef FPGA
    // FPGA: run the app in MACHINE mode. The original `csrwi mstatus, 0` sets
    // MPP=0 -> mret drops to USER mode, which hangs on this CVA6 (U-mode + PMP
    // not effective here). All bring-up tests pass in M-mode, which needs no
    // PMP. So set MPP=Machine; FS/VS are enabled just below.
1:  li      t0, (3 << 11)      // MSTATUS_MPP = Machine
    csrw    mstatus, t0
#else
1:  csrwi mstatus, 0            // Verilator/riscv-tests: drop to User mode
#endif
    // Enable FP Extension
    li      t0, (MSTATUS_FS & (MSTATUS_FS >> 1))
    csrs    mstatus, t0
    csrwi   fcsr, 0
    // Enable Vector Extension
    li      t0, (MSTATUS_VS & (MSTATUS_VS >> 1))
    csrs    mstatus, t0
    // Enable the counters
    csrsi   mcounteren, 1
    csrsi   scounteren, 1
#ifndef FPGA
    // Call the RISC-V Test initialization function, if it exists
    // (skipped on FPGA: same AUIPC PCREL range issue as stvec_handler)
    la t0, rvtest_init
    beqz t0, 1f
    jalr t0
1:
#endif
    // Return to _eoc
    la      ra, _eoc
    // Call main
    la      t0, main
    csrw    mepc, t0
    mret

    .align 2
trap_vector:
#ifndef FPGA
    // Jump to the mtvec_handler, if it exists
    // (skipped on FPGA: same AUIPC PCREL range issue)
    la t5, mtvec_handler
    beqz t5, 1f
    jr t5
1:
#endif
#ifdef FPGA
    // K5-VGG: generically resume from an unserviced ecall (newlib syscall
    // stub, mcause=11) instead of treating it as fatal. newlib has ~50
    // possible syscall stubs (_open, _kill, _times, _link, ...) and there is
    // no way to enumerate up front which ones a future library addition
    // might pull in -- found by hand-fixing _fstat then _sbrk one at a time
    // (see util.c's block comment) before switching to this generic
    // approach. a0=-1 matches the negative-errno convention every one of
    // these reentrant wrapper stubs already checks for (confirmed in
    // _fstat_r's disassembly: `bgez a0` skips the errno-set path only on a
    // non-negative return -- -1 takes the same path a real kernel-rejected
    // syscall would). mepc+4 skips the ecall itself (always 4 bytes, no
    // compressed form) so execution resumes at the instruction right after
    // it, in the caller, as if the syscall had returned -1.
    //
    // Only mcause==11 (ECALL from M-mode) is intercepted here; every other
    // exception (illegal instruction, misaligned/access fault, page fault)
    // falls through to the diagnostic block and _fail exactly as before --
    // this does not make real hardware faults any less fatal.
    csrr t5, mcause
    li   t6, 11
    bne  t5, t6, .Lnot_ecall
    li   a0, -1
    csrr t5, mepc
    addi t5, t5, 4
    csrw mepc, t5
    mret
.Lnot_ecall:
#endif
    // ---- TRAP DIAGNOSTIC BLOCK ----------------------------------------
    // Record the trap unambiguously, FIRST, before anything else can go wrong.
    // Why the old handler was not enough: it only captured mcause and wrote it
    // to eoc_address_reg -- the same mailbox as a normal exit code, so a trap
    // returning 0 was indistinguishable from a clean exit, and the faulting PC
    // was lost entirely.
    // These go to dedicated addresses so (a) devmem can read them after the
    // _eoc flush, and (b) the ILA's GATED dbg_cva6_last_write shows
    // 0x50000050 -- a unique signature meaning "a trap happened" -- even when
    // _eoc is never reached and nothing is ever flushed to DDR.
    //   0x50000040 mcause[31:0]   0x50000044 mcause[63:32] (bit31 = interrupt)
    //   0x50000048 mepc[31:0]     (the faulting PC)
    //   0x5000004C mtval[31:0]    (faulting address / instruction)
    //   0x50000050 0x7A7A7A7A     (TRAP flag - watch for this on the ILA)
    li   t6, 0x50000040
    csrr t5, mcause
    sw   t5, 0(t6)
    srli t5, t5, 32
    sw   t5, 4(t6)
    csrr t5, mepc
    sw   t5, 8(t6)
    csrr t5, mtval
    sw   t5, 12(t6)
    li   t5, 0x7A7A7A7A
    sw   t5, 16(t6)
    fence rw, rw

    csrr a0, mcause
    j _fail

    .align 2
_eoc:
    la t0, eoc_address_reg
    sd a0, 0(t0)                  // publish return code (lands in LLC when cached)
// NO_LLC builds target the LlcNotBypass=0 bitstream, where cheshire_soc.sv's
// `if (Cfg.LlcOutConnect && Cfg.LlcNotBypass) : gen_llc` never instantiates
// axi_llc, so the config registers at 0x0300_1000 are UNMAPPED. Reading
// SET_ASSO there returns garbage and the CFG_FLUSH poll below spins forever --
// a hang that proves nothing about the design under test. There is also nothing
// to flush: with no write-back LLC the stores already went straight to DDR.
#if defined(FPGA) && !defined(NO_LLC)
    // FPGA: the LLC is write-back (LlcNotBypass=1). Flush ALL ways so the return
    // code, the DDR console buffer, and any results are written back to PS-DDR,
    // where the ARM host reads them via /dev/mem. axi_llc regs @ AmLlc=0x0300_1000:
    //   CFG_FLUSH_LOW +0x8, CFG_FLUSH_HIGH +0xc, COMMIT_CFG +0x10,
    //   FLUSHED_LOW +0x18, SET_ASSO_LOW +0x28 (num ways).
    li   t0, 0x03001000
    lw   t4, 0x28(t0)            // number of ways (set-associativity)
    li   t1, 1
    sll  t1, t1, t4
    addi t1, t1, -1             // flush mask = (1<<ways)-1   (0xff for 8 ways)
    sw   t1, 0x8(t0)            // CFG_FLUSH_LOW  = mask
    sw   x0, 0xc(t0)            // CFG_FLUSH_HIGH = 0
    li   t2, 1
    sw   t2, 0x10(t0)           // COMMIT_CFG.commit = 1  -> start flush
    // Poll CFG_FLUSH (0x8), NOT FLUSHED (0x18).
    // FLUSHED is a live state bitmap ("ways with no valid tags"), not a
    // completion flag: axi_llc_config.sv FsmEndFlush overwrites it with
    // cfg_spm (=0 here) when the LAST way finishes, so for 8 ways it walks
    // 0x01,0x03,...,0x7f,0 and NEVER equals the 0xff mask -> the old
    // `bne t3,t1,1b` spun forever even though the flush had completed.
    // CFG_FLUSH is the real completion signal: software writes it in FsmIdle
    // (cfg_flush_en=0 there), hardware holds it during the flush, and clears
    // it to 0 in FsmEndFlush/FsmInitFlush when the FSM returns to idle.
    // COMMIT_CFG cannot be used -- it self-clears when the flush STARTS.
1:  lw   t3, 0x8(t0)            // poll CFG_FLUSH_LOW
    bnez t3, 1b                 // hardware clears it when the flush completes
#endif
2:  j    2b                    // halt; everything is now resident in PS-DDR

    .align 2
_fail:
    la t0, eoc_address_reg
    sd a0, 0(t0)
    jal x0, _eoc               // -> flush (if FPGA) + halt

.section .data
