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
opTriBoxBlur_premul(struct image32* restrict Rimage,
                    const struct image32* restrict Simage,
                    uint32_t r);

int
main(int argc, char *argv[])
{
    struct image32 Simage;
    struct image32 Rimage;
    struct image32 REFimage;

    png32_decode(&Simage, "../res/cliff.png");
    image32_alloc(&Rimage, Simage.xsize, Simage.ysize);
    image32_alloc(&REFimage, Simage.xsize, Simage.ysize);

    opTriBoxBlur_premul_ref(&REFimage, &Simage, 5);

    printf("%p %dx%d\n", Simage.data, Simage.xsize, Simage.ysize);
    printf("%p %dx%d\n", Rimage.data, Rimage.xsize, Rimage.ysize);
    printf("%p %dx%d\n", REFimage.data, REFimage.xsize, REFimage.ysize);

    png32_encode(&REFimage, "./_out.png");
    image32_free(&Simage);
    image32_free(&Rimage);
    image32_free(&REFimage);
    return 0;
}
