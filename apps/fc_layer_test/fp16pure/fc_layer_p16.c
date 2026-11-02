//
// fp16pure/fc_layer_p16.c — compile-time symbol-rename shim.
//
// fc_layer16only/kernel/fc_layer.c exports fc_op_forward / fc_op_backward /
// calloc_fc_weights / ... with the same names as the mixed-precision fp16/ tree.
// Both trees are linked into the same fc_layer_test binary, so we rename every
// colliding public symbol to a _p16-suffixed variant before the compiler sees
// the implementation file.
//
// The #define trick works because:
//   - struct tags are renamed within this TU only (no ODR clash across TUs)
//   - function names become _p16 extern symbols in the object file
//   - static functions / static variables are TU-local anyway
//
#define fc_op                     fc_op_p16
#define fc_op_forward             fc_op_forward_p16
#define fc_op_backward            fc_op_backward_p16
#define fc_op_backward_input_only fc_op_backward_input_only_p16
#define calloc_fc_weights         calloc_fc_weights_p16
#define free_fc_weights           free_fc_weights_p16
#define calloc_fc_dweights        calloc_fc_dweights_p16
#define free_fc_dweights          free_fc_dweights_p16
#define load_fc_weights           load_fc_weights_p16
#define save_fc_weights           save_fc_weights_p16

#include "../../fc_layer16only/kernel/fc_layer.c"
