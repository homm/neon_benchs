#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>

#define REF _ref
#include "impl.native.c"
#undef REF

extern void
opSourceOver_premul(uint8_t* restrict Rrgba,
                    const uint8_t* restrict Srgba,
                    const uint8_t* restrict Drgba, size_t len);


uint8_t *
alloc_premul(size_t len)
{
    uint8_t *pixels = malloc(len * 4);
    for (size_t i = 0; i < len*4; i += 4) {
        pixels[i + 3] = rand() % 256;
        pixels[i + 0] = ((rand() % 256) * pixels[i + 3]) / 255;
        pixels[i + 1] = ((rand() % 256) * pixels[i + 3]) / 255;
        pixels[i + 2] = ((rand() % 256) * pixels[i + 3]) / 255;
    }
    return pixels;
}


int
main(int argc, char *argv[])
{
    size_t len = 1000;
    uint8_t *Srgba = alloc_premul(len);
    uint8_t *Drgba = alloc_premul(len);
    uint8_t *Rrgba = malloc(len * 4);
    uint8_t *REFrgba = malloc(len * 4);

    for (size_t j = 0; j < 4; j ++) {
        struct timeval tval_before, tval_after, tval_result;
        opSourceOver_premul_ref(REFrgba, Srgba, Drgba, len);
        
        for (size_t i = 0; i < 100 * 1000; i ++) {
            opSourceOver_premul(Rrgba, Srgba, Drgba, len);
        }
        gettimeofday(&tval_before, NULL);
        for (size_t i = 0; i < 100 * 1000; i ++) {
            opSourceOver_premul(Rrgba, Srgba, Drgba, len);
        }
        gettimeofday(&tval_after, NULL);
        timersub(&tval_after, &tval_before, &tval_result);

        printf("Time elapsed: %ld.%06ld s\n",
            (long int)tval_result.tv_sec, (long int)tval_result.tv_usec);

        for (size_t i = 0; i < len; i++) {
            if (((uint32_t *)REFrgba)[i] != ((uint32_t *)Rrgba)[i]) {
                printf("Err on pos %zu: %#08x %#08x\n", i, ((uint32_t *)REFrgba)[i], ((uint32_t *)Rrgba)[i]);
                break;
            }
        }
    }

    free(Srgba);
    free(Drgba);
    free(Rrgba);
    free(REFrgba);
    return 0;
}
