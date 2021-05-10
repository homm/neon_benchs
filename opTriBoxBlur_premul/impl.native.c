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
    uint32_t X1div = (uint32_t) (1 << 24) / d;
    uint32_t X2div = (uint32_t) (1 << 16) / d;
    uint32_t X3div = X2div;
    pixel128 X1;
    pixel128 X2;
    pixel128 X3;
    pixel64 b[r_mask + 1];
    pixel64 c[r_mask + 1];
    uint32_t lastx = Simg->xsize - 1;

    for (int y = 0; y < Simg->ysize; y += 1) {
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

        b[0].r = (uint16_t) ((X1.r * X1div) >> 16);
        b[0].g = (uint16_t) ((X1.g * X1div) >> 16);
        b[0].b = (uint16_t) ((X1.b * X1div) >> 16);
        b[0].a = (uint16_t) ((X1.a * X1div) >> 16);
        X2.r = b[0].r * (r + 1);
        X2.g = b[0].g * (r + 1);
        X2.b = b[0].b * (r + 1);
        X2.a = b[0].a * (r + 1);
        for (int x = 1; x <= r; x += 1) {
            X1.r += sdata[x + r].r - sdata[0].r;
            X1.g += sdata[x + r].g - sdata[0].g;
            X1.b += sdata[x + r].b - sdata[0].b;
            X1.a += sdata[x + r].a - sdata[0].a;
            b[x].r = (uint16_t) ((X1.r * X1div) >> 16);
            b[x].g = (uint16_t) ((X1.g * X1div) >> 16);
            b[x].b = (uint16_t) ((X1.b * X1div) >> 16);
            b[x].a = (uint16_t) ((X1.a * X1div) >> 16);
            X2.r += b[x].r;
            X2.g += b[x].g;
            X2.b += b[x].b;
            X2.a += b[x].a;
        }

        c[0].r = (uint16_t) ((X2.r * X2div) >> 16);
        c[0].g = (uint16_t) ((X2.g * X2div) >> 16);
        c[0].b = (uint16_t) ((X2.b * X2div) >> 16);
        c[0].a = (uint16_t) ((X2.a * X2div) >> 16);
        X3.r = c[0].r * (r + 2);
        X3.g = c[0].g * (r + 2);
        X3.b = c[0].b * (r + 2);
        X3.a = c[0].a * (r + 2);
        for (int x = 1; x < r; x += 1) {
            X1.r += sdata[x + r2].r - sdata[x - 1].r;
            X1.g += sdata[x + r2].g - sdata[x - 1].g;
            X1.b += sdata[x + r2].b - sdata[x - 1].b;
            X1.a += sdata[x + r2].a - sdata[x - 1].a;
            b[x + r].r = (uint16_t) ((X1.r * X1div) >> 16);
            b[x + r].g = (uint16_t) ((X1.g * X1div) >> 16);
            b[x + r].b = (uint16_t) ((X1.b * X1div) >> 16);
            b[x + r].a = (uint16_t) ((X1.a * X1div) >> 16);
            X2.r += b[x + r].r - b[0].r;
            X2.g += b[x + r].g - b[0].g;
            X2.b += b[x + r].b - b[0].b;
            X2.a += b[x + r].a - b[0].a;
            c[x].r = (uint16_t) ((X2.r * X2div) >> 16);
            c[x].g = (uint16_t) ((X2.g * X2div) >> 16);
            c[x].b = (uint16_t) ((X2.b * X2div) >> 16);
            c[x].a = (uint16_t) ((X2.a * X2div) >> 16);
            X3.r += c[x].r;
            X3.g += c[x].g;
            X3.b += c[x].b;
            X3.a += c[x].a;
        }

        b[-1 & r_mask] = b[0];
        for (int x = 0; x <= r; x += 1) {
            c[(x - r - 1) & r_mask] = c[0];
        }
        for (int x = 0; x < Simg->xsize - r3; x += 1) {
            pixel64 last_b, last_c;
            X1.r += sdata[x + r3].r - sdata[x + r - 1].r;
            X1.g += sdata[x + r3].g - sdata[x + r - 1].g;
            X1.b += sdata[x + r3].b - sdata[x + r - 1].b;
            X1.a += sdata[x + r3].a - sdata[x + r - 1].a;
            last_b.r = b[(x + r2) & r_mask].r = (uint16_t) ((X1.r * X1div) >> 16);
            last_b.g = b[(x + r2) & r_mask].g = (uint16_t) ((X1.g * X1div) >> 16);
            last_b.b = b[(x + r2) & r_mask].b = (uint16_t) ((X1.b * X1div) >> 16);
            last_b.a = b[(x + r2) & r_mask].a = (uint16_t) ((X1.a * X1div) >> 16);
            X2.r += last_b.r - b[(x - 1) & r_mask].r;
            X2.g += last_b.g - b[(x - 1) & r_mask].g;
            X2.b += last_b.b - b[(x - 1) & r_mask].b;
            X2.a += last_b.a - b[(x - 1) & r_mask].a;
            last_c.r = c[(x + r) & r_mask].r = (uint16_t) ((X2.r * X2div) >> 16);
            last_c.g = c[(x + r) & r_mask].g = (uint16_t) ((X2.g * X2div) >> 16);
            last_c.b = c[(x + r) & r_mask].b = (uint16_t) ((X2.b * X2div) >> 16);
            last_c.a = c[(x + r) & r_mask].a = (uint16_t) ((X2.a * X2div) >> 16);
            X3.r += last_c.r - c[(x - r - 1) & r_mask].r;
            X3.g += last_c.g - c[(x - r - 1) & r_mask].g;
            X3.b += last_c.b - c[(x - r - 1) & r_mask].b;
            X3.a += last_c.a - c[(x - r - 1) & r_mask].a;

            *rdata = (pixel32){
                (uint8_t) ((X3.r * X3div) >> 24),
                (uint8_t) ((X3.g * X3div) >> 24),
                (uint8_t) ((X3.b * X3div) >> 24),
                (uint8_t) ((X3.a * X3div) >> 24)
            };
            rdata += Rimg->xsize;
        }

        for (int x = Simg->xsize - r3; x < Simg->xsize - r2; x += 1) {
            pixel64 last_b, last_c;
            X1.r += sdata[lastx].r - sdata[x + r - 1].r;
            X1.g += sdata[lastx].g - sdata[x + r - 1].g;
            X1.b += sdata[lastx].b - sdata[x + r - 1].b;
            X1.a += sdata[lastx].a - sdata[x + r - 1].a;
            last_b.r = b[(x + r2) & r_mask].r = (uint16_t) ((X1.r * X1div) >> 16);
            last_b.g = b[(x + r2) & r_mask].g = (uint16_t) ((X1.g * X1div) >> 16);
            last_b.b = b[(x + r2) & r_mask].b = (uint16_t) ((X1.b * X1div) >> 16);
            last_b.a = b[(x + r2) & r_mask].a = (uint16_t) ((X1.a * X1div) >> 16);
            X2.r += last_b.r - b[(x - 1) & r_mask].r;
            X2.g += last_b.g - b[(x - 1) & r_mask].g;
            X2.b += last_b.b - b[(x - 1) & r_mask].b;
            X2.a += last_b.a - b[(x - 1) & r_mask].a;
            last_c.r = c[(x + r) & r_mask].r = (uint16_t) ((X2.r * X2div) >> 16);
            last_c.g = c[(x + r) & r_mask].g = (uint16_t) ((X2.g * X2div) >> 16);
            last_c.b = c[(x + r) & r_mask].b = (uint16_t) ((X2.b * X2div) >> 16);
            last_c.a = c[(x + r) & r_mask].a = (uint16_t) ((X2.a * X2div) >> 16);
            X3.r += last_c.r - c[(x - r - 1) & r_mask].r;
            X3.g += last_c.g - c[(x - r - 1) & r_mask].g;
            X3.b += last_c.b - c[(x - r - 1) & r_mask].b;
            X3.a += last_c.a - c[(x - r - 1) & r_mask].a;

            *rdata = (pixel32){
                (uint8_t) ((X3.r * X3div) >> 24),
                (uint8_t) ((X3.g * X3div) >> 24),
                (uint8_t) ((X3.b * X3div) >> 24),
                (uint8_t) ((X3.a * X3div) >> 24)
            };
            rdata += Rimg->xsize;
        }

        for (int x = Simg->xsize - r2; x < Simg->xsize - r; x += 1) {
            pixel64 last_c;
            X2.r += b[lastx & r_mask].r - b[(x - 1) & r_mask].r;
            X2.g += b[lastx & r_mask].g - b[(x - 1) & r_mask].g;
            X2.b += b[lastx & r_mask].b - b[(x - 1) & r_mask].b;
            X2.a += b[lastx & r_mask].a - b[(x - 1) & r_mask].a;
            last_c.r = c[(x + r) & r_mask].r = (uint16_t) ((X2.r * X2div) >> 16);
            last_c.g = c[(x + r) & r_mask].g = (uint16_t) ((X2.g * X2div) >> 16);
            last_c.b = c[(x + r) & r_mask].b = (uint16_t) ((X2.b * X2div) >> 16);
            last_c.a = c[(x + r) & r_mask].a = (uint16_t) ((X2.a * X2div) >> 16);
            X3.r += last_c.r - c[(x - r - 1) & r_mask].r;
            X3.g += last_c.g - c[(x - r - 1) & r_mask].g;
            X3.b += last_c.b - c[(x - r - 1) & r_mask].b;
            X3.a += last_c.a - c[(x - r - 1) & r_mask].a;

            *rdata = (pixel32){
                (uint8_t) ((X3.r * X3div) >> 24),
                (uint8_t) ((X3.g * X3div) >> 24),
                (uint8_t) ((X3.b * X3div) >> 24),
                (uint8_t) ((X3.a * X3div) >> 24)
            };
            rdata += Rimg->xsize;
        }

        for (int x = Simg->xsize - r; x < Simg->xsize; x += 1) {
            X3.r += c[lastx & r_mask].r - c[(x - r - 1) & r_mask].r;
            X3.g += c[lastx & r_mask].g - c[(x - r - 1) & r_mask].g;
            X3.b += c[lastx & r_mask].b - c[(x - r - 1) & r_mask].b;
            X3.a += c[lastx & r_mask].a - c[(x - r - 1) & r_mask].a;

            *rdata = (pixel32){
                (uint8_t) ((X3.r * X3div) >> 24),
                (uint8_t) ((X3.g * X3div) >> 24),
                (uint8_t) ((X3.b * X3div) >> 24),
                (uint8_t) ((X3.a * X3div) >> 24)
            };
            rdata += Rimg->xsize;
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
