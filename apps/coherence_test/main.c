// Copyright 2026 ETH Zurich and University of Bologna.
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

int main() {
    // Αρχικοποίηση πίνακα με 0
    // Χρησιμοποιούμε volatile για να αναγκάσουμε τον compiler να διαβάσει από τη μνήμη στο τέλος
    volatile float data[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    
    float scalar_add_val = 5.0f;
    float vector_add_val = 10.0f;

    printf_("Πριν το πείραμα: data[0] = %f\n", data[0]);

    // Το μπλοκ Assembly που ζήτησες με την ακριβή σειρά
    asm volatile (
        "vsetvli zero, %[vl], e32, m1, ta, ma \n\t"
        
        // 1. Vector Load: Ο Ara ρουφάει τα δεδομένα (όλα 0) στον v8
        "vle32.v v8, (%[ptr]) \n\t"
        
        // 2. Scalar Load: Ο CVA6 διαβάζει το πρώτο ψηφίο (0) στον fa5
        "flw fa5, 0(%[ptr]) \n\t"
        
        // 3. Scalar Add: Ο CVA6 προσθέτει το 5 (0 + 5 = 5)
        "fadd.s fa5, fa5, %[s_val] \n\t"
        
        
        
        // 5. Vector Add: Ο Ara προσθέτει το 10 στον καταχωρητή v8 (0 + 10 = 10)
        "vfadd.vf v8, v8, %[v_val] \n\t"

        // 4. Scalar Store: Ο CVA6 γράφει το 5 πίσω στη μνήμη (data[0])
        "fsw fa5, 0(%[ptr]) \n\t"
        
        // 6. Vector Store: Ο Ara γράφει όλον τον v8 πίσω στη μνήμη
        "vse32.v v8, (%[ptr]) \n\t"
        
        // Clobbers & Inputs
        : 
        : [vl] "r" (4), [ptr] "r" (data), [s_val] "f" (scalar_add_val), [v_val] "f" (vector_add_val)
        : "v8", "fa5", "memory"
    );

    // Τι θα τυπώσει άραγε;
    printf_("Μετά το πείραμα: data[0] = %f\n", data[0]);
    printf_("Μετά το πείραμα: data[1] = %f\n", data[1]);

    return 0;
}
