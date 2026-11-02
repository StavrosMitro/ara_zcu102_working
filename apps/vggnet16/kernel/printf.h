#pragma once
// Native (GCC/Linux) shim: map printf_ to standard printf/fprintf
#include <stdio.h>
#include <stdlib.h>
#define printf_(...) printf(__VA_ARGS__)
