#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>

#include "../image.h"

#define REF _ref
#include "impl.native.c"
#undef REF

extern void
opTriBoxBlur_premul(image32* restrict Rimage,
                    image32* restrict INTimage,
                    const image32* restrict Simage,
                    uint32_t r);

int
main(int argc, char *argv[])
{
    image32 Simage;
    image32 Rimage;
    image32 REFimage;
    image32 INTimage;

    png32_decode(&Simage, "../res/cliff.png");
    image32_alloc(&Rimage, Simage.xsize, Simage.ysize);
    image32_alloc(&REFimage, Simage.xsize, Simage.ysize);
    image32_alloc(&INTimage, Simage.ysize, Simage.xsize);

    opTriBoxBlur_premul_ref(&REFimage, &INTimage, &Simage, 5);

    printf("%p %dx%d\n", Simage.data, Simage.xsize, Simage.ysize);
    printf("%p %dx%d\n", Rimage.data, Rimage.xsize, Rimage.ysize);
    printf("%p %dx%d\n", REFimage.data, REFimage.xsize, REFimage.ysize);

    for (size_t j = 0; j < 4; j ++) {
        struct timeval tval_before, tval_after, tval_result;
        gettimeofday(&tval_before, NULL);
        for (size_t i = 0; i < 10; i ++) {
            opTriBoxBlur_premul(&Rimage, &INTimage, &Simage, 5);
        }
        gettimeofday(&tval_after, NULL);
        timersub(&tval_after, &tval_before, &tval_result);

        printf("Time elapsed: %ld.%06ld s\n",
            (long int)tval_result.tv_sec, (long int)tval_result.tv_usec);

        for (size_t y = 0; y < Rimage.ysize; y++) {
            uint32_t* rdata = (uint32_t*) Rimage.data + Rimage.xsize * y;
            uint32_t* refdata = (uint32_t*) REFimage.data + REFimage.xsize * y;
            for (size_t x = 0; x < Rimage.xsize; x++) {
                if (rdata[x] != refdata[x]) {
                    printf("Err on pos %zu,%zu: %#08x %#08x\n", x, y, rdata[x], refdata[x]);
                    goto end_check;
                }
            }
        }
        end_check: ;
    }

    png32_encode(&REFimage, "./_out.png");
    image32_free(&Simage);
    image32_free(&Rimage);
    image32_free(&REFimage);
    image32_free(&INTimage);
    return 0;
}
