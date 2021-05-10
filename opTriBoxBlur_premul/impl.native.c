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
    image32* restrict Rimg,
    const image32* restrict Simg, uint32_t r)
{
    uint32_t d = r * 2 + 1;
    uint32_t r2 = r * 2, r3 = r * 3;
    uint32_t r_mask = all_bits_mask(d);
    float rev_d = 1.0 / d;
    float rev_d256 = rev_d / 256.0;
    float rev_256d = 256.0 * rev_d;
    pixel128 X1;
    pixel128 X2;
    pixel128 X3;
    pixel64 b[r_mask + 1];
    pixel64 c[r_mask + 1];

    for (int y = 0; y < Simg->ysize - 1; y += 1) {
        pixel32* sdata = (pixel32*) (Simg->data + Simg->next_line * y);
        pixel32* rdata = (pixel32*) Rimg->data + y;
        
        X1.r = sdata[0].r * (r + 1);
        X1.g = sdata[0].g * (r + 1);
        X1.b = sdata[0].b * (r + 1);
        X1.a = sdata[0].a * (r + 1);
        for (int x = 1; x <= r; x += 1) {
            X1.r += sdata[x].r;
            X1.g += sdata[x].g;
            X1.b += sdata[x].b;
            X1.a += sdata[x].a;
        }

        b[0].r = (uint32_t) (X1.r * rev_256d);
        b[0].g = (uint32_t) (X1.g * rev_256d);
        b[0].b = (uint32_t) (X1.b * rev_256d);
        b[0].a = (uint32_t) (X1.a * rev_256d);
        X2.r = b[0].r * (r + 1);
        X2.g = b[0].g * (r + 1);
        X2.b = b[0].b * (r + 1);
        X2.a = b[0].a * (r + 1);
        for (int x = 1; x <= r; x += 1) {
            X1.r += sdata[x + r].r - sdata[0].r;
            X1.g += sdata[x + r].g - sdata[0].g;
            X1.b += sdata[x + r].b - sdata[0].b;
            X1.a += sdata[x + r].a - sdata[0].a;
            b[x].r = (uint32_t) (X1.r * rev_256d);
            b[x].g = (uint32_t) (X1.g * rev_256d);
            b[x].b = (uint32_t) (X1.b * rev_256d);
            b[x].a = (uint32_t) (X1.a * rev_256d);
            X2.r += b[x].r;
            X2.g += b[x].g;
            X2.b += b[x].b;
            X2.a += b[x].a;
        }

        c[0].r = (uint32_t) (X2.r * rev_d);
        c[0].g = (uint32_t) (X2.g * rev_d);
        c[0].b = (uint32_t) (X2.b * rev_d);
        c[0].a = (uint32_t) (X2.a * rev_d);
        X3.r = c[0].r * (r + 2);
        X3.g = c[0].g * (r + 2);
        X3.b = c[0].b * (r + 2);
        X3.a = c[0].a * (r + 2);
        for (int x = 1; x < r; x += 1) {
            X1.r += sdata[x + r2].r - sdata[x - 1].r;
            X1.g += sdata[x + r2].g - sdata[x - 1].g;
            X1.b += sdata[x + r2].b - sdata[x - 1].b;
            X1.a += sdata[x + r2].a - sdata[x - 1].a;
            b[x + r].r = (uint32_t) (X1.r * rev_256d);
            b[x + r].g = (uint32_t) (X1.g * rev_256d);
            b[x + r].b = (uint32_t) (X1.b * rev_256d);
            b[x + r].a = (uint32_t) (X1.a * rev_256d);
            X2.r += b[x + r].r - b[0].r;
            X2.g += b[x + r].g - b[0].g;
            X2.b += b[x + r].b - b[0].b;
            X2.a += b[x + r].a - b[0].a;
            c[x].r = (uint32_t) (X2.r * rev_d);
            c[x].g = (uint32_t) (X2.g * rev_d);
            c[x].b = (uint32_t) (X2.b * rev_d);
            c[x].a = (uint32_t) (X2.a * rev_d);
            X3.r += c[x].r;
            X3.g += c[x].g;
            X3.b += c[x].b;
            X3.a += c[x].a;
        }

        b[-1 & r_mask] = b[0];
        for (int x = 0; x <= r; x += 1) {
            c[(x - r - 1) & r_mask] = c[0];
        }
        for (int x = 0; x < Simg->xsize; x += 1) {
            pixel64 last_b, last_c;
            X1.r += sdata[x + r3].r - sdata[x + r - 1].r;
            X1.g += sdata[x + r3].g - sdata[x + r - 1].g;
            X1.b += sdata[x + r3].b - sdata[x + r - 1].b;
            X1.a += sdata[x + r3].a - sdata[x + r - 1].a;
            last_b.r = b[(x + r2) & r_mask].r = (uint32_t) (X1.r * rev_256d);
            last_b.g = b[(x + r2) & r_mask].g = (uint32_t) (X1.g * rev_256d);
            last_b.b = b[(x + r2) & r_mask].b = (uint32_t) (X1.b * rev_256d);
            last_b.a = b[(x + r2) & r_mask].a = (uint32_t) (X1.a * rev_256d);
            X2.r += last_b.r - b[(x - 1) & r_mask].r;
            X2.g += last_b.g - b[(x - 1) & r_mask].g;
            X2.b += last_b.b - b[(x - 1) & r_mask].b;
            X2.a += last_b.a - b[(x - 1) & r_mask].a;
            last_c.r = c[(x + r) & r_mask].r = (uint32_t) (X2.r * rev_d);
            last_c.g = c[(x + r) & r_mask].g = (uint32_t) (X2.g * rev_d);
            last_c.b = c[(x + r) & r_mask].b = (uint32_t) (X2.b * rev_d);
            last_c.a = c[(x + r) & r_mask].a = (uint32_t) (X2.a * rev_d);
            X3.r += last_c.r - c[(x - r - 1) & r_mask].r;
            X3.g += last_c.g - c[(x - r - 1) & r_mask].g;
            X3.b += last_c.b - c[(x - r - 1) & r_mask].b;
            X3.a += last_c.a - c[(x - r - 1) & r_mask].a;

            rdata[x * Rimg->xsize] = (pixel32){
                (uint8_t) (X3.r * rev_d256),
                (uint8_t) (X3.g * rev_d256),
                (uint8_t) (X3.b * rev_d256),
                (uint8_t) (X3.a * rev_d256)
            };
        }
    }
}

extern void
DEFINE_REF(opTriBoxBlur_premul)(
    image32* restrict Rimage,
    image32* restrict INTimage,
    const image32* restrict Simage, uint32_t r)
{
    opTriBoxBlur_premul_horz(INTimage, Simage, r);
    opTriBoxBlur_premul_horz(Rimage, INTimage, r);
}
