#include <assert.h>
#include <stdint.h>
#include <stddef.h>
#include <math.h>
#include "../image.h"

#ifndef REF
    #define REF
#endif

#define PASTER(x, y) x ## y
#define EVALUATOR(x, y)  PASTER(x, y)
#define DEFINE_REF(name) EVALUATOR(name, REF)


static void
opTriBoxBlur_horz_larger(
    image32* restrict Rimg,
    const image32* restrict Simg, uint32_t r)
{
    // Max r for this implementation is 32767.
    uint32_t d = r * 2 + 1;
    uint32_t r2 = r * 2, r3 = r * 3;
    uint32_t r_mask = all_bits_mask(d);
    // Each accumulator (Xn) consists of:
    // * 8 bits — source data
    // * from 1 to 16 bits - for accumulating (max d = 65535)
    // * not less than 8 bits remains for integer division
    uint32_t X1div = (1 << 24) / (float) d + 0.5;
    uint32_t X2div = X1div;
    uint32_t X3div = X1div;
    pixel128 X1;
    pixel128 X2;
    pixel128 X3;
    pixel32 b[r_mask + 1];
    pixel32 c[r_mask + 1];
    uint32_t lastx = Simg->xsize - 1;

    assert(r < (1 << 15));
    assert(Simg->xsize >= r3);

    for (size_t y = 0; y < Simg->ysize; y += 1) {
        pixel32* sdata = (pixel32*) Simg->data + Simg->xsize * y;
        pixel32* rdata = (pixel32*) Rimg->data + y;
        
        X1.r = sdata[0].r * (r + 1) + (1<<23) / X1div;
        X1.g = sdata[0].g * (r + 1) + (1<<23) / X1div;
        X1.b = sdata[0].b * (r + 1) + (1<<23) / X1div;
        X1.a = sdata[0].a * (r + 1) + (1<<23) / X1div;
        for (size_t x = 1; x <= r; x += 1) {
            X1.r += sdata[x].r;
            X1.g += sdata[x].g;
            X1.b += sdata[x].b;
            X1.a += sdata[x].a;
        }

        b[0].r = (uint8_t) ((X1.r * X1div) >> 24);
        b[0].g = (uint8_t) ((X1.g * X1div) >> 24);
        b[0].b = (uint8_t) ((X1.b * X1div) >> 24);
        b[0].a = (uint8_t) ((X1.a * X1div) >> 24);
        X2.r = b[0].r * (r + 1) + (1<<23) / X2div;
        X2.g = b[0].g * (r + 1) + (1<<23) / X2div;
        X2.b = b[0].b * (r + 1) + (1<<23) / X2div;
        X2.a = b[0].a * (r + 1) + (1<<23) / X2div;
        for (size_t x = 1; x <= r; x += 1) {
            X1.r += sdata[x + r].r - sdata[0].r;
            X1.g += sdata[x + r].g - sdata[0].g;
            X1.b += sdata[x + r].b - sdata[0].b;
            X1.a += sdata[x + r].a - sdata[0].a;
            b[x].r = (uint8_t) ((X1.r * X1div) >> 24);
            b[x].g = (uint8_t) ((X1.g * X1div) >> 24);
            b[x].b = (uint8_t) ((X1.b * X1div) >> 24);
            b[x].a = (uint8_t) ((X1.a * X1div) >> 24);
            X2.r += b[x].r;
            X2.g += b[x].g;
            X2.b += b[x].b;
            X2.a += b[x].a;
        }

        c[0].r = (uint8_t) ((X2.r * X2div) >> 24);
        c[0].g = (uint8_t) ((X2.g * X2div) >> 24);
        c[0].b = (uint8_t) ((X2.b * X2div) >> 24);
        c[0].a = (uint8_t) ((X2.a * X2div) >> 24);
        X3.r = c[0].r * (r + 2) + (1<<23) / X3div;
        X3.g = c[0].g * (r + 2) + (1<<23) / X3div;
        X3.b = c[0].b * (r + 2) + (1<<23) / X3div;
        X3.a = c[0].a * (r + 2) + (1<<23) / X3div;
        for (size_t x = 1; x < r; x += 1) {
            X1.r += sdata[x + r2].r - sdata[x - 1].r;
            X1.g += sdata[x + r2].g - sdata[x - 1].g;
            X1.b += sdata[x + r2].b - sdata[x - 1].b;
            X1.a += sdata[x + r2].a - sdata[x - 1].a;
            b[x + r].r = (uint8_t) ((X1.r * X1div) >> 24);
            b[x + r].g = (uint8_t) ((X1.g * X1div) >> 24);
            b[x + r].b = (uint8_t) ((X1.b * X1div) >> 24);
            b[x + r].a = (uint8_t) ((X1.a * X1div) >> 24);
            X2.r += b[x + r].r - b[0].r;
            X2.g += b[x + r].g - b[0].g;
            X2.b += b[x + r].b - b[0].b;
            X2.a += b[x + r].a - b[0].a;
            c[x].r = (uint8_t) ((X2.r * X2div) >> 24);
            c[x].g = (uint8_t) ((X2.g * X2div) >> 24);
            c[x].b = (uint8_t) ((X2.b * X2div) >> 24);
            c[x].a = (uint8_t) ((X2.a * X2div) >> 24);
            X3.r += c[x].r;
            X3.g += c[x].g;
            X3.b += c[x].b;
            X3.a += c[x].a;
        }

        b[-1 & r_mask] = b[0];
        for (size_t x = 0; x <= r; x += 1) {
            c[(x - r - 1) & r_mask] = c[0];
        }
        for (size_t x = 0; x < Simg->xsize - r3; x += 1) {
            pixel32 last_b, last_c;
            X1.r += sdata[x + r3].r - sdata[x + r - 1].r;
            X1.g += sdata[x + r3].g - sdata[x + r - 1].g;
            X1.b += sdata[x + r3].b - sdata[x + r - 1].b;
            X1.a += sdata[x + r3].a - sdata[x + r - 1].a;
            last_b.r = b[(x + r2) & r_mask].r = (uint8_t) ((X1.r * X1div) >> 24);
            last_b.g = b[(x + r2) & r_mask].g = (uint8_t) ((X1.g * X1div) >> 24);
            last_b.b = b[(x + r2) & r_mask].b = (uint8_t) ((X1.b * X1div) >> 24);
            last_b.a = b[(x + r2) & r_mask].a = (uint8_t) ((X1.a * X1div) >> 24);
            X2.r += last_b.r - b[(x - 1) & r_mask].r;
            X2.g += last_b.g - b[(x - 1) & r_mask].g;
            X2.b += last_b.b - b[(x - 1) & r_mask].b;
            X2.a += last_b.a - b[(x - 1) & r_mask].a;
            last_c.r = c[(x + r) & r_mask].r = (uint8_t) ((X2.r * X2div) >> 24);
            last_c.g = c[(x + r) & r_mask].g = (uint8_t) ((X2.g * X2div) >> 24);
            last_c.b = c[(x + r) & r_mask].b = (uint8_t) ((X2.b * X2div) >> 24);
            last_c.a = c[(x + r) & r_mask].a = (uint8_t) ((X2.a * X2div) >> 24);
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

        for (size_t x = Simg->xsize - r3; x < Simg->xsize - r2; x += 1) {
            pixel32 last_b, last_c;
            X1.r += sdata[lastx].r - sdata[x + r - 1].r;
            X1.g += sdata[lastx].g - sdata[x + r - 1].g;
            X1.b += sdata[lastx].b - sdata[x + r - 1].b;
            X1.a += sdata[lastx].a - sdata[x + r - 1].a;
            last_b.r = b[(x + r2) & r_mask].r = (uint8_t) ((X1.r * X1div) >> 24);
            last_b.g = b[(x + r2) & r_mask].g = (uint8_t) ((X1.g * X1div) >> 24);
            last_b.b = b[(x + r2) & r_mask].b = (uint8_t) ((X1.b * X1div) >> 24);
            last_b.a = b[(x + r2) & r_mask].a = (uint8_t) ((X1.a * X1div) >> 24);
            X2.r += last_b.r - b[(x - 1) & r_mask].r;
            X2.g += last_b.g - b[(x - 1) & r_mask].g;
            X2.b += last_b.b - b[(x - 1) & r_mask].b;
            X2.a += last_b.a - b[(x - 1) & r_mask].a;
            last_c.r = c[(x + r) & r_mask].r = (uint8_t) ((X2.r * X2div) >> 24);
            last_c.g = c[(x + r) & r_mask].g = (uint8_t) ((X2.g * X2div) >> 24);
            last_c.b = c[(x + r) & r_mask].b = (uint8_t) ((X2.b * X2div) >> 24);
            last_c.a = c[(x + r) & r_mask].a = (uint8_t) ((X2.a * X2div) >> 24);
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

        for (size_t x = Simg->xsize - r2; x < Simg->xsize - r; x += 1) {
            pixel32 last_c;
            X2.r += b[lastx & r_mask].r - b[(x - 1) & r_mask].r;
            X2.g += b[lastx & r_mask].g - b[(x - 1) & r_mask].g;
            X2.b += b[lastx & r_mask].b - b[(x - 1) & r_mask].b;
            X2.a += b[lastx & r_mask].a - b[(x - 1) & r_mask].a;
            last_c.r = c[(x + r) & r_mask].r = (uint8_t) ((X2.r * X2div) >> 24);
            last_c.g = c[(x + r) & r_mask].g = (uint8_t) ((X2.g * X2div) >> 24);
            last_c.b = c[(x + r) & r_mask].b = (uint8_t) ((X2.b * X2div) >> 24);
            last_c.a = c[(x + r) & r_mask].a = (uint8_t) ((X2.a * X2div) >> 24);
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

        for (size_t x = Simg->xsize - r; x < Simg->xsize; x += 1) {
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


static void
opTriBoxBlur_horz_smallr(
    image32* restrict Rimg,
    const image32* restrict Simg, float floatR)
{
    // Max floatR for this implementation is 128 (excluded).
    uint32_t r = floatR;
    uint32_t d = r * 2 + 1;
    uint32_t r2 = r * 2, r3 = r * 3;
    // We need two extra slots for writing even if E1div = 0
    uint32_t r_mask = all_bits_mask(d + 2);
    // Each accumulator (Xn) consists of:
    // * 8 bits — source data
    // * from 1 to 8 bits - for accumulating (max d = 255)
    // * not less than 8 bits remains for integer division
    uint16_t X1div = (1 << 16) / (floatR * 2 + 1) + 0.5;
    uint16_t X2div = X1div, X3div = X1div;
    // uint16_t E1div = ((1 << 16) - d * X1div) / 2 + 0.5;
    uint16_t E1div = ((1 << 16) - d * X1div) + 0.5;
    uint16_t E2div = E1div, E3div = E1div;
    pixel64 X1;
    pixel64 X2;
    pixel64 X3;
    pixel32 b[r_mask + 1];
    pixel32 c[r_mask + 1];
    size_t lastx = Simg->xsize - 1;

    assert(floatR < 128);
    assert(Simg->xsize >= r3);

    for (size_t y = 0; y < Simg->ysize; y += 1) {
        pixel32* sdata = (pixel32*) Simg->data + Simg->xsize * y;
        pixel32* rdata = (pixel32*) Rimg->data + y;
        
        X1.r = sdata[0].r * (r + 1) + (1<<15) / X1div;
        X1.g = sdata[0].g * (r + 1) + (1<<15) / X1div;
        X1.b = sdata[0].b * (r + 1) + (1<<15) / X1div;
        X1.a = sdata[0].a * (r + 1) + (1<<15) / X1div;
        for (size_t x = 1; x <= r; x += 1) {
            X1.r += sdata[x].r;
            X1.g += sdata[x].g;
            X1.b += sdata[x].b;
            X1.a += sdata[x].a;
        }

                                            // prev
        b[0].r = (uint8_t) ((X1.r * X1div + (sdata[0].r) * E1div) >> 16);
        b[0].g = (uint8_t) ((X1.g * X1div + (sdata[0].g) * E1div) >> 16);
        b[0].b = (uint8_t) ((X1.b * X1div + (sdata[0].b) * E1div) >> 16);
        b[0].a = (uint8_t) ((X1.a * X1div + (sdata[0].a) * E1div) >> 16);
        X2.r = b[0].r * (r + 1) + (1<<15) / X2div;
        X2.g = b[0].g * (r + 1) + (1<<15) / X2div;
        X2.b = b[0].b * (r + 1) + (1<<15) / X2div;
        X2.a = b[0].a * (r + 1) + (1<<15) / X2div;
        for (size_t x = 1; x <= r; x += 1) {
            X1.r += sdata[x + r].r - sdata[0].r;
            X1.g += sdata[x + r].g - sdata[0].g;
            X1.b += sdata[x + r].b - sdata[0].b;
            X1.a += sdata[x + r].a - sdata[0].a;
                                                // prev
            b[x].r = (uint8_t) ((X1.r * X1div + (sdata[0].r) * E1div) >> 16);
            b[x].g = (uint8_t) ((X1.g * X1div + (sdata[0].g) * E1div) >> 16);
            b[x].b = (uint8_t) ((X1.b * X1div + (sdata[0].b) * E1div) >> 16);
            b[x].a = (uint8_t) ((X1.a * X1div + (sdata[0].a) * E1div) >> 16);
            X2.r += b[x].r;
            X2.g += b[x].g;
            X2.b += b[x].b;
            X2.a += b[x].a;
        }

                                            // prev
        c[0].r = (uint8_t) ((X2.r * X2div + (b[0].r) * E2div) >> 16);
        c[0].g = (uint8_t) ((X2.g * X2div + (b[0].g) * E2div) >> 16);
        c[0].b = (uint8_t) ((X2.b * X2div + (b[0].b) * E2div) >> 16);
        c[0].a = (uint8_t) ((X2.a * X2div + (b[0].a) * E2div) >> 16);
        X3.r = c[0].r * (r + 2) + (1<<15) / X3div;
        X3.g = c[0].g * (r + 2) + (1<<15) / X3div;
        X3.b = c[0].b * (r + 2) + (1<<15) / X3div;
        X3.a = c[0].a * (r + 2) + (1<<15) / X3div;
        for (size_t x = 1; x < r; x += 1) {
            X1.r += sdata[x + r2].r - sdata[x - 1].r;
            X1.g += sdata[x + r2].g - sdata[x - 1].g;
            X1.b += sdata[x + r2].b - sdata[x - 1].b;
            X1.a += sdata[x + r2].a - sdata[x - 1].a;
                                                    // next
            b[x + r].r = (uint8_t) ((X1.r * X1div + (sdata[x + r2 + 1].r) * E1div) >> 16);
            b[x + r].g = (uint8_t) ((X1.g * X1div + (sdata[x + r2 + 1].g) * E1div) >> 16);
            b[x + r].b = (uint8_t) ((X1.b * X1div + (sdata[x + r2 + 1].b) * E1div) >> 16);
            b[x + r].a = (uint8_t) ((X1.a * X1div + (sdata[x + r2 + 1].a) * E1div) >> 16);
            X2.r += b[x + r].r - b[0].r;
            X2.g += b[x + r].g - b[0].g;
            X2.b += b[x + r].b - b[0].b;
            X2.a += b[x + r].a - b[0].a;
                                                // prev
            c[x].r = (uint8_t) ((X2.r * X2div + (b[0].r) * E2div) >> 16);
            c[x].g = (uint8_t) ((X2.g * X2div + (b[0].g) * E2div) >> 16);
            c[x].b = (uint8_t) ((X2.b * X2div + (b[0].b) * E2div) >> 16);
            c[x].a = (uint8_t) ((X2.a * X2div + (b[0].a) * E2div) >> 16);
            X3.r += c[x].r;
            X3.g += c[x].g;
            X3.b += c[x].b;
            X3.a += c[x].a;
        }

        b[-1 & r_mask] = b[0];
        for (size_t x = 0; x <= r; x += 1) {
            c[(x - r - 1) & r_mask] = c[0];
        }
        for (size_t x = 0; x < Simg->xsize - r3; x += 1) {
            pixel32 last_b, last_c;
            X1.r += sdata[x + r3].r - sdata[x + r - 1].r;
            X1.g += sdata[x + r3].g - sdata[x + r - 1].g;
            X1.b += sdata[x + r3].b - sdata[x + r - 1].b;
            X1.a += sdata[x + r3].a - sdata[x + r - 1].a;
                                                                           // prev
            last_b.r = b[(x + r2) & r_mask].r = (uint8_t) ((X1.r * X1div + (sdata[x + r - 2].r) * E1div) >> 16);
            last_b.g = b[(x + r2) & r_mask].g = (uint8_t) ((X1.g * X1div + (sdata[x + r - 2].g) * E1div) >> 16);
            last_b.b = b[(x + r2) & r_mask].b = (uint8_t) ((X1.b * X1div + (sdata[x + r - 2].b) * E1div) >> 16);
            last_b.a = b[(x + r2) & r_mask].a = (uint8_t) ((X1.a * X1div + (sdata[x + r - 2].a) * E1div) >> 16);
            X2.r += last_b.r - b[(x - 1) & r_mask].r;
            X2.g += last_b.g - b[(x - 1) & r_mask].g;
            X2.b += last_b.b - b[(x - 1) & r_mask].b;
            X2.a += last_b.a - b[(x - 1) & r_mask].a;
                                                                          // prev
            last_c.r = c[(x + r) & r_mask].r = (uint8_t) ((X2.r * X2div + (b[(x - 2) & r_mask].r) * E2div) >> 16);
            last_c.g = c[(x + r) & r_mask].g = (uint8_t) ((X2.g * X2div + (b[(x - 2) & r_mask].g) * E2div) >> 16);
            last_c.b = c[(x + r) & r_mask].b = (uint8_t) ((X2.b * X2div + (b[(x - 2) & r_mask].b) * E2div) >> 16);
            last_c.a = c[(x + r) & r_mask].a = (uint8_t) ((X2.a * X2div + (b[(x - 2) & r_mask].a) * E2div) >> 16);
            X3.r += last_c.r - c[(x - r - 1) & r_mask].r;
            X3.g += last_c.g - c[(x - r - 1) & r_mask].g;
            X3.b += last_c.b - c[(x - r - 1) & r_mask].b;
            X3.a += last_c.a - c[(x - r - 1) & r_mask].a;

            *rdata = (pixel32){
                                           // prev
                (uint8_t) ((X3.r * X3div + (c[(x - r - 2) & r_mask].r) * E3div) >> 16),
                (uint8_t) ((X3.g * X3div + (c[(x - r - 2) & r_mask].g) * E3div) >> 16),
                (uint8_t) ((X3.b * X3div + (c[(x - r - 2) & r_mask].b) * E3div) >> 16),
                (uint8_t) ((X3.a * X3div + (c[(x - r - 2) & r_mask].a) * E3div) >> 16)
            };
            rdata += Rimg->xsize;
        }

        for (size_t x = Simg->xsize - r3; x < Simg->xsize - r2; x += 1) {
            pixel32 last_b, last_c;
            X1.r += sdata[lastx].r - sdata[x + r - 1].r;
            X1.g += sdata[lastx].g - sdata[x + r - 1].g;
            X1.b += sdata[lastx].b - sdata[x + r - 1].b;
            X1.a += sdata[lastx].a - sdata[x + r - 1].a;
                                                                           // prev
            last_b.r = b[(x + r2) & r_mask].r = (uint8_t) ((X1.r * X1div + (sdata[x + r - 2].r) * E1div) >> 16);
            last_b.g = b[(x + r2) & r_mask].g = (uint8_t) ((X1.g * X1div + (sdata[x + r - 2].g) * E1div) >> 16);
            last_b.b = b[(x + r2) & r_mask].b = (uint8_t) ((X1.b * X1div + (sdata[x + r - 2].b) * E1div) >> 16);
            last_b.a = b[(x + r2) & r_mask].a = (uint8_t) ((X1.a * X1div + (sdata[x + r - 2].a) * E1div) >> 16);
            X2.r += last_b.r - b[(x - 1) & r_mask].r;
            X2.g += last_b.g - b[(x - 1) & r_mask].g;
            X2.b += last_b.b - b[(x - 1) & r_mask].b;
            X2.a += last_b.a - b[(x - 1) & r_mask].a;
                                                                          // prev
            last_c.r = c[(x + r) & r_mask].r = (uint8_t) ((X2.r * X2div + (b[(x - 2) & r_mask].r) * E2div) >> 16);
            last_c.g = c[(x + r) & r_mask].g = (uint8_t) ((X2.g * X2div + (b[(x - 2) & r_mask].g) * E2div) >> 16);
            last_c.b = c[(x + r) & r_mask].b = (uint8_t) ((X2.b * X2div + (b[(x - 2) & r_mask].b) * E2div) >> 16);
            last_c.a = c[(x + r) & r_mask].a = (uint8_t) ((X2.a * X2div + (b[(x - 2) & r_mask].a) * E2div) >> 16);
            X3.r += last_c.r - c[(x - r - 1) & r_mask].r;
            X3.g += last_c.g - c[(x - r - 1) & r_mask].g;
            X3.b += last_c.b - c[(x - r - 1) & r_mask].b;
            X3.a += last_c.a - c[(x - r - 1) & r_mask].a;

            *rdata = (pixel32){
                                           // prev
                (uint8_t) ((X3.r * X3div + (c[(x - r - 2) & r_mask].r) * E3div) >> 16),
                (uint8_t) ((X3.g * X3div + (c[(x - r - 2) & r_mask].g) * E3div) >> 16),
                (uint8_t) ((X3.b * X3div + (c[(x - r - 2) & r_mask].b) * E3div) >> 16),
                (uint8_t) ((X3.a * X3div + (c[(x - r - 2) & r_mask].a) * E3div) >> 16)
            };
            rdata += Rimg->xsize;
        }

        for (size_t x = Simg->xsize - r2; x < Simg->xsize - r; x += 1) {
            pixel32 last_c;
            X2.r += b[lastx & r_mask].r - b[(x - 1) & r_mask].r;
            X2.g += b[lastx & r_mask].g - b[(x - 1) & r_mask].g;
            X2.b += b[lastx & r_mask].b - b[(x - 1) & r_mask].b;
            X2.a += b[lastx & r_mask].a - b[(x - 1) & r_mask].a;
                                                                          // prev
            last_c.r = c[(x + r) & r_mask].r = (uint8_t) ((X2.r * X2div + (b[(x - 2) & r_mask].r) * E2div) >> 16);
            last_c.g = c[(x + r) & r_mask].g = (uint8_t) ((X2.g * X2div + (b[(x - 2) & r_mask].g) * E2div) >> 16);
            last_c.b = c[(x + r) & r_mask].b = (uint8_t) ((X2.b * X2div + (b[(x - 2) & r_mask].b) * E2div) >> 16);
            last_c.a = c[(x + r) & r_mask].a = (uint8_t) ((X2.a * X2div + (b[(x - 2) & r_mask].a) * E2div) >> 16);
            X3.r += last_c.r - c[(x - r - 1) & r_mask].r;
            X3.g += last_c.g - c[(x - r - 1) & r_mask].g;
            X3.b += last_c.b - c[(x - r - 1) & r_mask].b;
            X3.a += last_c.a - c[(x - r - 1) & r_mask].a;

            *rdata = (pixel32){
                                           // prev
                (uint8_t) ((X3.r * X3div + (c[(x - r - 2) & r_mask].r) * E3div) >> 16),
                (uint8_t) ((X3.g * X3div + (c[(x - r - 2) & r_mask].g) * E3div) >> 16),
                (uint8_t) ((X3.b * X3div + (c[(x - r - 2) & r_mask].b) * E3div) >> 16),
                (uint8_t) ((X3.a * X3div + (c[(x - r - 2) & r_mask].a) * E3div) >> 16)
            };
            rdata += Rimg->xsize;
        }

        for (size_t x = Simg->xsize - r; x < Simg->xsize; x += 1) {
            X3.r += c[lastx & r_mask].r - c[(x - r - 1) & r_mask].r;
            X3.g += c[lastx & r_mask].g - c[(x - r - 1) & r_mask].g;
            X3.b += c[lastx & r_mask].b - c[(x - r - 1) & r_mask].b;
            X3.a += c[lastx & r_mask].a - c[(x - r - 1) & r_mask].a;

            *rdata = (pixel32){
                                           // prev
                (uint8_t) ((X3.r * X3div + (c[(x - r - 2) & r_mask].r) * E3div) >> 16),
                (uint8_t) ((X3.g * X3div + (c[(x - r - 2) & r_mask].g) * E3div) >> 16),
                (uint8_t) ((X3.b * X3div + (c[(x - r - 2) & r_mask].b) * E3div) >> 16),
                (uint8_t) ((X3.a * X3div + (c[(x - r - 2) & r_mask].a) * E3div) >> 16)
            };
            rdata += Rimg->xsize;
        }
    }
}


extern void
DEFINE_REF(opTriBoxBlur_premul)(
    image32* restrict Rimage,
    image32* restrict INTimage,
    const image32* restrict Simage, float r)
{
    if (r < 128) {
        opTriBoxBlur_horz_smallr(INTimage, Simage, r);
        opTriBoxBlur_horz_smallr(Rimage, INTimage, r);
    } else {
        opTriBoxBlur_horz_larger(INTimage, Simage, r + 0.5);
        opTriBoxBlur_horz_larger(Rimage, INTimage, r + 0.5);
    }
}
