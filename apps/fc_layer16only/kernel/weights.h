#ifndef WEIGHTS_H
#define WEIGHTS_H

#include "alexnet.h"   // FC_INPUT_UNITS / FC_OUTPUT_UNITS (single source of truth)

// From weights.c — bounds derive from the FC dimension knob in alexnet.h so that
// sizeof(fc1_weights_32) (used by the shape check in main.c) tracks the real size.
extern _Float16 fc1_weights_32[FC_INPUT_UNITS * FC_OUTPUT_UNITS];
extern _Float16 fc1_bias_32[FC_OUTPUT_UNITS];

// From the assembly file (.incbin)
extern const _Float16 test_inputs_32[];
extern const _Float16 test_targets_32[];
extern const int test_labels_32[];

#endif