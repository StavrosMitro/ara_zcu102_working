.section .rodata

.global test_inputs_32
.align 4
test_inputs_32:
    .incbin "generated_data/inputs.bin"

.global test_targets_32
.align 4
test_targets_32:
    .incbin "generated_data/targets.bin"

.global test_labels_32
.align 4
test_labels_32:
    .incbin "generated_data/labels.bin"

.global cifar10_data_32
.align 4
cifar10_data_32:
    .incbin "generated_data/inputs.bin"

.global cifar10_labels_32
.align 4
cifar10_labels_32:
    .incbin "generated_data/labels.bin"