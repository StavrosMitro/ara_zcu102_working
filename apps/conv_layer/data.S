.section .rodata

.global test_inputs
.align 4
test_inputs:
    .incbin "generated_data/inputs.bin"

.global test_targets
.align 4
test_targets:
    .incbin "generated_data/targets.bin"

.global test_labels
.align 4
test_labels:
    .incbin "generated_data/labels.bin"

.global cifar10_data
.align 4
cifar10_data:
    .incbin "generated_data/inputs.bin"

.global cifar10_labels
.align 4
cifar10_labels:
    .incbin "generated_data/labels.bin"