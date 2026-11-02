// dma_test needs no data blob, but the .fpga16 link rule expects
// <app>/data.S.o.fpga16, so provide a minimal translation unit.
.section .rodata
.global dma_test_data_marker
.align 4
dma_test_data_marker:
    .word 0xD1A00000
