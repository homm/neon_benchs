#include <assert.h>
#include <stdint.h>
#include <stddef.h>
#include <emmintrin.h>
#include <tmmintrin.h>
#include <smmintrin.h>
#include <xmmintrin.h>
#include "../image.h"


static __m128i inline
mm_cvtepu8_epi32(void *ptr) {
    return _mm_cvtepu8_epi32(_mm_cvtsi32_si128(*(uint32_t *) ptr));
}

void inline
mm_storeu_si32(void *ptr, __m128i v) {
    *((int32_t *) ptr) = _mm_cvtsi128_si32(v);
}


static void
opTriBoxBlur_premul_horz(
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
    __m128i X1div = _mm_set1_epi32((1 << 24) / (double) d + 0.5);
    __m128i X2div = X1div;
    __m128i X3div = X1div;
    __m128i b23 = _mm_set1_epi32(1 << 23);
    __m128i storemask = _mm_set_epi8(0,0,0,0,0,0,0,0,0,0,0,0,15,11,7,3);
    __m128i X1;
    __m128i X2;
    __m128i X3;
    __m128i temp;
    __m128i b[r_mask + 1];
    __m128i c[r_mask + 1];
    size_t xsize = Simg->xsize;
    size_t lastx = Simg->xsize - 1;

    #define LOAD(x) mm_cvtepu8_epi32(&sdata[(x) * xsize])

    assert(r < (1 << 15));
    assert(Simg->ysize >= r3);

    for (size_t y = 0; y < Simg->xsize; y += 1) {
        pixel32* sdata = (pixel32*) Simg->data + y;
        pixel32* rdata = (pixel32*) Rimg->data + Rimg->xsize * y;
        
        // X1.r = sdata[0].r * (r + 1);
        X1 = _mm_mullo_epi32(LOAD(0), _mm_set1_epi32(r + 1));
        for (size_t x = 1; x <= r; x += 1) {
            // X1.r += sdata[x].r;
            X1 = _mm_add_epi32(X1, LOAD(x));
        }

        // b[0].r = (uint8_t) ((X1.r * X1div + (1 << 23)) >> 24);
        b[0] = _mm_srli_epi32(_mm_add_epi32(
            _mm_mullo_epi32(X1, X1div), b23), 24);
        // X2.r = b[0].r * (r + 1);
        X2 = _mm_mullo_epi32(b[0], _mm_set1_epi32(r + 1));
        for (size_t x = 1; x <= r; x += 1) {
            // X1.r += sdata[x + r].r - sdata[0].r;
            X1 = _mm_add_epi32(LOAD(x + r),
                _mm_sub_epi32(X1, LOAD(0)));
            // b[x].r = (uint8_t) ((X1.r * X1div + (1 << 23)) >> 24);
            b[x] = _mm_srli_epi32(_mm_add_epi32(
                _mm_mullo_epi32(X1, X1div), b23), 24);
            // X2.r += b[x].r;
            X2 = _mm_add_epi32(X2, b[x]);
        }

        // c[0].r = (uint8_t) ((X2.r * X2div + (1 << 23)) >> 24);
        c[0] = _mm_srli_epi32(_mm_add_epi32(
            _mm_mullo_epi32(X2, X2div), b23), 24);
        // X3.r = c[0].r * (r + 2);
        X3 = _mm_mullo_epi32(c[0], _mm_set1_epi32(r + 2));
        for (size_t x = 1; x < r; x += 1) {
            // X1.r += sdata[x + r2].r - sdata[x - 1].r;
            X1 = _mm_add_epi32(LOAD(x + r2),
                _mm_sub_epi32(X1, LOAD(x - 1)));
            // b[x + r].r = (uint8_t) ((X1.r * X1div + (1 << 23)) >> 24);
            b[x + r] = _mm_srli_epi32(_mm_add_epi32(
                _mm_mullo_epi32(X1, X1div), b23), 24);
            // X2.r += b[x + r].r - b[0].r;
            X2 = _mm_add_epi32(b[x + r],
                _mm_sub_epi32(X2, b[0]));
            // c[x].r = (uint8_t) ((X2.r * X2div + (1 << 23)) >> 24);
            c[x] = _mm_srli_epi32(_mm_add_epi32(
                _mm_mullo_epi32(X2, X2div), b23), 24);
            // X3.r += c[x].r;
            X3 = _mm_add_epi32(X3, c[x]);
        }

        b[-1 & r_mask] = b[0];
        for (size_t x = 0; x <= r; x += 1) {
            c[(x - r - 1) & r_mask] = c[0];
        }
        for (size_t x = 0; x < Simg->ysize - r3; x += 1) {
            // pixel32 last_b, last_c;
            // X1.r += sdata[x + r3].r - sdata[x + r - 1].r;
            X1 = _mm_add_epi32(LOAD(x + r3),
                _mm_sub_epi32(X1, LOAD(x + r - 1)));
            // last_b.r = b[(x + r2) & r_mask].r = (uint8_t) ((X1.r * X1div + (1 << 23)) >> 24);
            b[(x + r2) & r_mask] = _mm_srli_epi32(_mm_add_epi32(
                _mm_mullo_epi32(X1, X1div), b23), 24);
            // X2.r += last_b.r - b[(x - 1) & r_mask].r;
            X2 = _mm_add_epi32(b[(x + r2) & r_mask],
                _mm_sub_epi32(X2, b[(x - 1) & r_mask]));
            // last_c.r = c[(x + r) & r_mask].r = (uint8_t) ((X2.r * X2div + (1 << 23)) >> 24);
            c[(x + r) & r_mask] = _mm_srli_epi32(_mm_add_epi32(
                _mm_mullo_epi32(X2, X2div), b23), 24);
            // X3.r += last_c.r - c[(x - r - 1) & r_mask].r;
            X3 = _mm_add_epi32(c[(x + r) & r_mask],
                _mm_sub_epi32(X3, c[(x - r - 1) & r_mask]));

            // *rdata.r = (uint8_t) ((X3.r * X3div + (1 << 23)) >> 24);
            temp = _mm_add_epi32(_mm_mullo_epi32(X3, X3div), b23);
            mm_storeu_si32(&rdata[x], _mm_shuffle_epi8(temp, storemask));
        }

        for (size_t x = Simg->ysize - r3; x < Simg->ysize - r2; x += 1) {
            // pixel32 last_b, last_c;
            // X1.r += sdata[lastx].r - sdata[x + r - 1].r;
            X1 = _mm_add_epi32(LOAD(lastx),
                _mm_sub_epi32(X1, LOAD(x + r - 1)));
            // last_b.r = b[(x + r2) & r_mask].r = (uint8_t) ((X1.r * X1div + (1 << 23)) >> 24);
            b[(x + r2) & r_mask] = _mm_srli_epi32(_mm_add_epi32(
                _mm_mullo_epi32(X1, X1div), b23), 24);
            // X2.r += last_b.r - b[(x - 1) & r_mask].r;
            X2 = _mm_add_epi32(b[(x + r2) & r_mask],
                _mm_sub_epi32(X2, b[(x - 1) & r_mask]));
            // last_c.r = c[(x + r) & r_mask].r = (uint8_t) ((X2.r * X2div + (1 << 23)) >> 24);
            c[(x + r) & r_mask] = _mm_srli_epi32(_mm_add_epi32(
                _mm_mullo_epi32(X2, X2div), b23), 24);
            // X3.r += last_c.r - c[(x - r - 1) & r_mask].r;
            X3 = _mm_add_epi32(c[(x + r) & r_mask],
                _mm_sub_epi32(X3, c[(x - r - 1) & r_mask]));

            // *rdata.r = (uint8_t) ((X3.r * X3div + (1 << 23)) >> 24);
            temp = _mm_add_epi32(_mm_mullo_epi32(X3, X3div), b23);
            mm_storeu_si32(&rdata[x], _mm_shuffle_epi8(temp, storemask));
        }

        for (size_t x = Simg->ysize - r2; x < Simg->ysize - r; x += 1) {
            // pixel32 last_c;
            // X2.r += b[lastx & r_mask].r - b[(x - 1) & r_mask].r;
            X2 = _mm_add_epi32(b[lastx & r_mask],
                _mm_sub_epi32(X2, b[(x - 1) & r_mask]));
            // last_c.r = c[(x + r) & r_mask].r = (uint8_t) ((X2.r * X2div + (1 << 23)) >> 24);
            c[(x + r) & r_mask] = _mm_srli_epi32(_mm_add_epi32(
                _mm_mullo_epi32(X2, X2div), b23), 24);
            // X3.r += last_c.r - c[(x - r - 1) & r_mask].r;
            X3 = _mm_add_epi32(c[(x + r) & r_mask],
                _mm_sub_epi32(X3, c[(x - r - 1) & r_mask]));

            // *rdata.r = (uint8_t) ((X3.r * X3div + (1 << 23)) >> 24);
            temp = _mm_add_epi32(_mm_mullo_epi32(X3, X3div), b23);
            mm_storeu_si32(&rdata[x], _mm_shuffle_epi8(temp, storemask));
        }

        for (size_t x = Simg->ysize - r; x < Simg->ysize; x += 1) {
            // X3.r += c[lastx & r_mask].r - c[(x - r - 1) & r_mask].r;
            X3 = _mm_add_epi32(c[lastx & r_mask],
                _mm_sub_epi32(X3, c[(x - r - 1) & r_mask]));

            // *rdata.r = (uint8_t) ((X3.r * X3div + (1 << 23)) >> 24);
            temp = _mm_add_epi32(_mm_mullo_epi32(X3, X3div), b23);
            mm_storeu_si32(&rdata[x], _mm_shuffle_epi8(temp, storemask));
        }
    }
}


extern void
opTriBoxBlur_premul(
    image32* restrict Rimage,
    image32* restrict INTimage,
    const image32* restrict Simage, uint32_t r)
{
    opTriBoxBlur_premul_horz(INTimage, Simage, r);
    opTriBoxBlur_premul_horz(Rimage, INTimage, r);
}
