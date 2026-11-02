/* FPGA (Cheshire) linker script — template; do not use directly.
   Build system copies this to link.fpga.ld; align_sections.sh then runs
   over it but is a NO-OP here — section alignment is pinned to ALIGN(16)
   below instead of being derived from nr_lanes (align_sections.sh still
   drives the Verilator common/link.ld). One .fpga image is therefore
   byte-identical on every bitstream in the L2/L4 sweep: the kernels are
   already VLA and nothing else in the image depends on the config.

   Key differences from arch.link.ld (Verilator):
     ORIGIN  = 0xC0000000  (LlcOutRegionStart — LLC scratchpad)
     LENGTH  = 0x00800000  (8 MB — adjust to match your LLC size)
     fake_uart             = 0x03002000  (Cheshire UART TX FIFO)

   crt0.S is shared with Verilator but three patterns required #ifdef FPGA:
     _stack_top            — la sp, _stack_top  replaces the Verilator
                             ld sp, 0(dram_end_address_reg) indirection
                             (no TB on FPGA to pre-load that register).
     eoc_address_reg       — crt0 writes the return code here and loops;
                             points to LLC scratchpad (harmless on FPGA).
     stvec_handler / rvtest_init / mtvec_handler weak-symbol LA calls are
     skipped on FPGA because undefined weak symbols resolve to 0x0, and
     the AUIPC PCREL offset from 0xC0000000 to 0x0 is -3 GB — beyond the
     ±2 GB PCREL range.  Verilator code at 0x80000000 is exactly at -2 GB,
     which just fits.

   event_trigger / hw_cnt_en_reg are aliased to LLC scratchpad so apps
   using runtime.h still link.  Adjust offsets if LLC < 8 MB.
*/

OUTPUT_ARCH("riscv")
ENTRY(_start)

MEMORY {
  LLC : ORIGIN = 0x40000000, LENGTH = 0x20000000
}

SECTIONS {
  .text : {
    *(.text.init)
    *(.text)
    *(.text.*)
  } > LLC

  /* Alignment PINNED to 16 (not 4*nr_lanes) so the linked .fpga image is
     config-independent: one binary for every L2/L4 bitstream. 16 = widest
     AXI beat in that sweep (4 lanes -> AxiDataWidth 128b -> 16 B) and is
     >= the 2-lane beat (8 B), so bulk vector/DMA reads still start on a
     whole beat on both. DO NOT raise to >= 32: .rodata : ALIGN(N) then
     re-aligns past .pad_rodata_guard's 16-mod-32 placement (see below) and
     puts .rodata back on a 32-B boundary — the 0x40004F60 cold-refill
     condition. An 8/16-lane part (beat 32/64 B) needs the guard reworked
     first. */
  .data : ALIGN(16) {
    *(.data)
    *(.data.*)
  } > LLC

  /* ---------------------------------------------------------------------
     GUARD GAP (2026-09-01) -- WORKAROUND, not a fix. The mechanism is not
     understood.
     Empirical: with .rodata starting at 0x40004F60 -- simultaneously the
     first .rodata address, 32-byte aligned (CVA6 DcacheLineWidth=256), and on
     the cache line right after _etext -- CVA6's cold refill returned that
     first halfword as 0x0000 while DRAM held the correct value (verified by
     ARM-side devmem before AND after the run). That single bad constant
     cascaded into 29 of 39 vec_test failures.
     Relinking so .rodata starts at 16 (mod 32) made all 29 disappear, and a
     1 KiB cold-refill probe over 32 distinct 32-byte lines is clean, so this
     is NOT general .rodata corruption -- it was that specific address.
     ALIGN(32) then +16 reproduces the known-good placement: .rodata begins
     16 bytes past a 32-byte boundary and at least 16 bytes past _etext.
     NOTE: alignment and _etext-adjacency were changed together in that
     experiment, so which one matters is still unknown. Remove this only when
     the root cause is found.
     --------------------------------------------------------------------- */
  /* Guard section. NB: a bare `. = ALIGN(32); . = . + 16;` here is a NO-OP --
     with `> LLC` region allocation the region allocator overrides the location
     counter. It must be a real output section, as below. */
  .pad_rodata_guard : ALIGN(32) { . = . + 16; } > LLC
  .rodata  : ALIGN(16) { *(.rodata .rodata.* .gnu.linkonce.r.*) } > LLC
  .rodata1 : ALIGN(16) { *(.rodata1) } > LLC
  .sdata2  : ALIGN(16) {
    *(.sdata2 .sdata2.* .gnu.linkonce.s2.*)
  } > LLC

  .sdata   : ALIGN(16) {
    __global_pointer$ = . + 0x800;
    *(.srodata.cst16) *(.srodata.cst8) *(.srodata.cst4) *(.srodata.cst2) *(.srodata .srodata.*)
    *(.sdata .sdata.* .gnu.linkonce.s.*)
  } > LLC

  .bss : ALIGN(16) {
    __bss_start = .;
    *(.bss)
    *(.sbss .sbss.* .sbss2 .sbss2.* .gnu.linkonce.sb2.*);
    __bss_end = .;
  } > LLC

  .l2 : ALIGN(16) {
    *(.l2)
    timer = .;
    . = . + 0x8;
    l2_alloc_base = ALIGN(16);
  } > LLC

  .comment : ALIGN(16) { *(.comment) } > LLC

  /* Top of stack: crt0.S does  la sp, _stack_top  under #ifdef FPGA.
     = ORIGIN(LLC) + LENGTH(LLC); adjust if your LLC is smaller than 8 MB. */
  _stack_top = 0x60000000;

  /* crt0.S writes the return code here and loops.  Points to LLC scratchpad
     so the write is harmless on FPGA. */
  eoc_address_reg = 0x5FFFF010;

  /* Cheshire UART TX FIFO — used by serial.c under #ifdef FPGA. */
  fake_uart = 0x03002000;

  /* Verilator TB register aliases → LLC scratchpad (harmless on FPGA). */
  event_trigger = 0x5FFFF000;
  hw_cnt_en_reg = 0x5FFFF008;
}
