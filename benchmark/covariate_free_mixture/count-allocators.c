#include <stdint.h>
#include <stddef.h>
extern void *__libc_malloc(size_t);
extern void *__libc_calloc(size_t, size_t);
extern void *__libc_realloc(void *, size_t);
extern void __libc_free(void *);
static _Thread_local unsigned armed;
/* malloc, calloc, realloc, non-null free, then requested bytes by allocator.
 * A realloc call need not move storage. These are call counts, not a live
 * memory measure; requested realloc bytes are not newly allocated bytes. */
static _Thread_local uint64_t counters[7];
void *malloc(size_t n) {
    if (armed) { ++counters[0]; counters[4] += n; }
    return __libc_malloc(n);
}
void *calloc(size_t n, size_t size) {
    if (armed) { ++counters[1]; counters[5] += n * size; }
    return __libc_calloc(n, size);
}
void *realloc(void *p, size_t n) {
    if (armed) { ++counters[2]; counters[6] += n; }
    return __libc_realloc(p, n);
}
void free(void *p) {
    if (armed && p) ++counters[3];
    __libc_free(p);
}
void begin_count(void) {
    for (unsigned i = 0; i < 7; ++i) counters[i] = 0;
    armed = 1;
}
uint64_t end_count(void) {
    armed = 0;
    return ((uint64_t)(uint32_t)counters[0] << 32) | (uint32_t)counters[3];
}
void end_counts(uint64_t *out) {
    armed = 0;
    for (unsigned i = 0; i < 7; ++i) out[i] = counters[i];
}
