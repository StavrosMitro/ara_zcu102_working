#include <stdint.h>

// #define UART_MMIO_ADDR 0xC0000000UL

#ifdef SPIKE
static volatile uint64_t spike_magic_mem[8] __attribute__((aligned(64)));
static volatile char spike_out_char;
#elif !defined(FPGA)
extern char fake_uart;
#endif


#define UART_BASE    0x03002000UL
#define UART_STATUS  (*(volatile uint32_t *)(UART_BASE + 0x10))
#define UART_WDATA   (*(volatile uint32_t *)(UART_BASE + 0x1C))
#define UART_TXFULL  (1 << 5)


void _putchar(char character) {
#ifdef SPIKE
    extern volatile uint64_t tohost;
    extern volatile uint64_t fromhost;
    //same code as syscall func in syscalls.c
    spike_out_char = character;
    spike_magic_mem[0] = 64;                       // SYS_write
    spike_magic_mem[1] = 1;                        // stdout fd
    spike_magic_mem[2] = (uintptr_t)&spike_out_char;
    spike_magic_mem[3] = 1;                        // one byte

    __sync_synchronize();
    tohost = (uintptr_t)spike_magic_mem;
    while (fromhost == 0)
        ;
    fromhost = 0;
    __sync_synchronize();
#elif defined(FPGA)
    // DDR "console": write each char to a DDR buffer; the ARM reads it back via
    // /dev/mem and forwards it to /dev/ttyPS1 (-> laptop /dev/ttyUSB1).
    // Rationale: Cheshire's real UART (0x03002000) transmits to PS UART1 on EMIO,
    // but the prebuilt Linux routes UART1 to MIO -> nothing on ttyPS1. So we output
    // to DDR instead. LLC is bypassed + CVA6 L1 is write-through => stores are live
    // in DDR. Reset the counter once before each run:  devmem 0x50000004 32 0
    // Old (dead) UART path kept for reference:
    //   while (UART_STATUS & UART_TXFULL);
    //   UART_WDATA = (uint32_t)character;
    // --- BRING-UP INSTRUMENTATION (temporary): trace where _putchar dies ---
    *(volatile uint32_t *)0x50000074UL = 0xAA; __sync_synchronize();  // entered _putchar
    volatile uint32_t *dbg_len = (volatile uint32_t *)0x50000004UL; // byte count
    volatile uint8_t  *dbg_buf = (volatile uint8_t  *)0x50002000UL; // text buffer (16 MB)
    uint32_t i = *dbg_len;
    *(volatile uint32_t *)0x50000078UL = i; __sync_synchronize();     // read len OK (value i)
    dbg_buf[i] = (uint8_t)character;
    *(volatile uint32_t *)0x5000007CUL = 0xCC; __sync_synchronize();  // buffer store OK
    __sync_synchronize();
    *dbg_len = i + 1;
    *(volatile uint32_t *)0x50000080UL = 0xDD; __sync_synchronize();  // len store OK (before drain)
    // WRITE-PATH DRAIN (bring-up fix). The LLC->HP0->PS-DDR write path deadlocks
    // under rapid back-to-back stores: an identical char-output loop HANGS with no
    // spacing but COMPLETES with a per-char delay (proven on HW, test_deep_combo).
    // A `fence` alone does NOT help -- it orders but does not drain past the
    // write-through L1 (serial.c already had __sync_synchronize per char and printf
    // still hung). So spend real time letting the two stores above reach DDR before
    // the next char. printf here is only sparse logging (cycles/accuracy), so this
    // cost is irrelevant. Tune SERIAL_DRAIN if a longer/shorter drain is needed.
    __sync_synchronize();
    for (volatile int __drain = 0; __drain < 4000; __drain++) { /* let stores reach DDR */ }
#else
    fake_uart = character;
#endif
}