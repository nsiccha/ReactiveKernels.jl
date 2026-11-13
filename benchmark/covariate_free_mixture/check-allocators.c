#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
extern void begin_count(void);
extern void end_counts(uint64_t *);
int main(void) {
    uint64_t out[7];
    const uint64_t expected[7] = {1, 1, 1, 2, 17, 12, 33};
    begin_count();
    void *a = malloc(17);
    void *b = calloc(3, 4);
    a = realloc(a, 33);
    free(a);
    free(b);
    end_counts(out);
    for (unsigned i = 0; i < 7; ++i)
        if (out[i] != expected[i]) {
            fprintf(stderr, "counter[%u]=%llu expected=%llu\n", i,
                (unsigned long long)out[i], (unsigned long long)expected[i]);
            return 1;
        }
    puts("COUNTER_CHECK malloc=1 calloc=1 realloc=1 free=2 requested_bytes=17/12/33");
    return 0;
}
