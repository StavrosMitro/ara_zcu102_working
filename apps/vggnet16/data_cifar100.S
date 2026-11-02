.section .rodata

.global cifar10_data
.align 4
cifar10_data:
    .incbin "kernel/cifar100_data.bin"

.global cifar10_offsets
.align 4
cifar10_offsets:
    .incbin "kernel/cifar100_offsets.bin"

.global cifar10_labels
.align 4
cifar10_labels:
    .incbin "kernel/cifar100_labels.bin"
