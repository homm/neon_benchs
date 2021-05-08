#include <stdint.h>
#include <stddef.h>
#include "../image.h"

#ifndef REF
    #define REF
#endif

#define PASTER(x, y) x ## y
#define EVALUATOR(x, y)  PASTER(x, y)
#define DEFINE_REF(name) EVALUATOR(name, REF)


static void
opTriBoxBlur_premul_horz(
    struct image32* restrict Rimage,
    const struct image32* restrict Simage, uint32_t r)
{
    for (int y = 0; y < Simage->ysize; y++) {
        /* code */
    }
}

extern void
DEFINE_REF(opTriBoxBlur_premul)(
    struct image32* restrict Rimage,
    const struct image32* restrict Simage, uint32_t r)
{
    opTriBoxBlur_premul_horz(Rimage, Simage, r);
}
