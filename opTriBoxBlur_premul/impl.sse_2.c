#include <assert.h>
#include <stdint.h>
#include <stddef.h>
#include <emmintrin.h>
#include <tmmintrin.h>
#include <smmintrin.h>
#include <xmmintrin.h>
#include "../image.h"


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
    __m128i X1div = _mm_set1_epi32((1 << 24) / (double) d + 0.5);
    __m128i X2div = X1div;
    __m128i X3div = X1div;
    __m128i b23 = _mm_set1_epi32(1 << 23);
    __m128i X1[4];
    __m128i X2[4];
    __m128i X3[4];
    __m128i temp[4];
    __m128i _temp0, _temp1;
    __m128i line[4];
    __m128i b[r_mask + 1];
    __m128i c[r_mask + 1];
    size_t xsize = Simg->xsize;
    size_t lastx = Simg->ysize - 1;

    #define LOADSDATA(x) \
        _temp0 = _mm_loadu_si128((__m128i*) &sdata[(x) * xsize]); \
        _temp1 = _mm_unpackhi_epi8(_temp0, _mm_setzero_si128()); \
        _temp0 = _mm_unpacklo_epi8(_temp0, _mm_setzero_si128());
    #define LOAD(src) \
        _temp0 = _mm_unpacklo_epi8((src), _mm_setzero_si128()); \
        _temp1 = _mm_unpackhi_epi8((src), _mm_setzero_si128());
    #define STORETEMP(dst) \
        (dst) = _mm_packus_epi16( \
            _mm_packus_epi32(temp[0], temp[1]), \
            _mm_packus_epi32(temp[2], temp[3]));
    #define GET0 _mm_unpacklo_epi16(_temp0, _mm_setzero_si128())
    #define GET1 _mm_unpackhi_epi16(_temp0, _mm_setzero_si128())
    #define GET2 _mm_unpacklo_epi16(_temp1, _mm_setzero_si128())
    #define GET3 _mm_unpackhi_epi16(_temp1, _mm_setzero_si128())

    assert(r < (1 << 15));
    assert(Simg->ysize >= r3);

    for (size_t y = 0; y < Simg->xsize; y += 4) {
        if (y > Simg->xsize - 4) {
            y = Simg->xsize - 4;
        }
        pixel32* sdata = (pixel32*) Simg->data + y;
        pixel32* rdata = (pixel32*) Rimg->data + Rimg->xsize * y;
        
        // X1.r = sdata[0].r * (r + 1);
        LOADSDATA(0);
        X1[0] = _mm_mullo_epi32(GET0, _mm_set1_epi32(r + 1));
        X1[1] = _mm_mullo_epi32(GET1, _mm_set1_epi32(r + 1));
        X1[2] = _mm_mullo_epi32(GET2, _mm_set1_epi32(r + 1));
        X1[3] = _mm_mullo_epi32(GET3, _mm_set1_epi32(r + 1));
        for (size_t x = 1; x <= r; x += 1) {
            // X1.r += sdata[x].r;
            LOADSDATA(x);
            X1[0] = _mm_add_epi32(X1[0], GET0);
            X1[1] = _mm_add_epi32(X1[1], GET1);
            X1[2] = _mm_add_epi32(X1[2], GET2);
            X1[3] = _mm_add_epi32(X1[3], GET3);
        }

        // b[0].r = (uint8_t) ((X1.r * X1div + (1 << 23)) >> 24);
        temp[0] = _mm_srli_epi32(_mm_add_epi32(_mm_mullo_epi32(X1[0], X1div), b23), 24);
        temp[1] = _mm_srli_epi32(_mm_add_epi32(_mm_mullo_epi32(X1[1], X1div), b23), 24);
        temp[2] = _mm_srli_epi32(_mm_add_epi32(_mm_mullo_epi32(X1[2], X1div), b23), 24);
        temp[3] = _mm_srli_epi32(_mm_add_epi32(_mm_mullo_epi32(X1[3], X1div), b23), 24);
        STORETEMP(b[0]);
        // X2.r = b[0].r * (r + 1);
        X2[0] = _mm_mullo_epi32(temp[0], _mm_set1_epi32(r + 1));
        X2[1] = _mm_mullo_epi32(temp[1], _mm_set1_epi32(r + 1));
        X2[2] = _mm_mullo_epi32(temp[2], _mm_set1_epi32(r + 1));
        X2[3] = _mm_mullo_epi32(temp[3], _mm_set1_epi32(r + 1));
        for (size_t x = 1; x <= r; x += 1) {
            // X1.r += sdata[x + r].r - sdata[0].r;
            LOADSDATA(0);
            X1[0] = _mm_sub_epi32(X1[0], GET0);
            X1[1] = _mm_sub_epi32(X1[1], GET1);
            X1[2] = _mm_sub_epi32(X1[2], GET2);
            X1[3] = _mm_sub_epi32(X1[3], GET3);
            LOADSDATA(x + r);
            X1[0] = _mm_add_epi32(X1[0], GET0);
            X1[1] = _mm_add_epi32(X1[1], GET1);
            X1[2] = _mm_add_epi32(X1[2], GET2);
            X1[3] = _mm_add_epi32(X1[3], GET3);
            // b[x].r = (uint8_t) ((X1.r * X1div + (1 << 23)) >> 24);
            temp[0] = _mm_srli_epi32(_mm_add_epi32(_mm_mullo_epi32(X1[0], X1div), b23), 24);
            temp[1] = _mm_srli_epi32(_mm_add_epi32(_mm_mullo_epi32(X1[1], X1div), b23), 24);
            temp[2] = _mm_srli_epi32(_mm_add_epi32(_mm_mullo_epi32(X1[2], X1div), b23), 24);
            temp[3] = _mm_srli_epi32(_mm_add_epi32(_mm_mullo_epi32(X1[3], X1div), b23), 24);
            STORETEMP(b[x]);
            // X2.r += b[x].r;
            X2[0] = _mm_add_epi32(X2[0], temp[0]);
            X2[1] = _mm_add_epi32(X2[1], temp[1]);
            X2[2] = _mm_add_epi32(X2[2], temp[2]);
            X2[3] = _mm_add_epi32(X2[3], temp[3]);
        }

        // c[0].r = (uint8_t) ((X2.r * X2div + (1 << 23)) >> 24);
        temp[0] = _mm_srli_epi32(_mm_add_epi32(_mm_mullo_epi32(X2[0], X2div), b23), 24);
        temp[1] = _mm_srli_epi32(_mm_add_epi32(_mm_mullo_epi32(X2[1], X2div), b23), 24);
        temp[2] = _mm_srli_epi32(_mm_add_epi32(_mm_mullo_epi32(X2[2], X2div), b23), 24);
        temp[3] = _mm_srli_epi32(_mm_add_epi32(_mm_mullo_epi32(X2[3], X2div), b23), 24);
        STORETEMP(c[0]);
        // X3.r = c[0].r * (r + 2);
        X3[0] = _mm_mullo_epi32(temp[0], _mm_set1_epi32(r + 2));
        X3[1] = _mm_mullo_epi32(temp[1], _mm_set1_epi32(r + 2));
        X3[2] = _mm_mullo_epi32(temp[2], _mm_set1_epi32(r + 2));
        X3[3] = _mm_mullo_epi32(temp[3], _mm_set1_epi32(r + 2));
        for (size_t x = 1; x < r; x += 1) {
            // X1.r += sdata[x + r2].r - sdata[x - 1].r;
            LOADSDATA(x - 1);
            X1[0] = _mm_sub_epi32(X1[0], GET0);
            X1[1] = _mm_sub_epi32(X1[1], GET1);
            X1[2] = _mm_sub_epi32(X1[2], GET2);
            X1[3] = _mm_sub_epi32(X1[3], GET3);
            LOADSDATA(x + r2);
            X1[0] = _mm_add_epi32(X1[0], GET0);
            X1[1] = _mm_add_epi32(X1[1], GET1);
            X1[2] = _mm_add_epi32(X1[2], GET2);
            X1[3] = _mm_add_epi32(X1[3], GET3);
            // b[x + r].r = (uint8_t) ((X1.r * X1div + (1 << 23)) >> 24);
            temp[0] = _mm_srli_epi32(_mm_add_epi32(_mm_mullo_epi32(X1[0], X1div), b23), 24);
            temp[1] = _mm_srli_epi32(_mm_add_epi32(_mm_mullo_epi32(X1[1], X1div), b23), 24);
            temp[2] = _mm_srli_epi32(_mm_add_epi32(_mm_mullo_epi32(X1[2], X1div), b23), 24);
            temp[3] = _mm_srli_epi32(_mm_add_epi32(_mm_mullo_epi32(X1[3], X1div), b23), 24);
            STORETEMP(b[x + r]);
            // X2.r += b[x + r].r - b[0].r;
            LOAD(b[0]);
            X2[0] = _mm_add_epi32(temp[0], _mm_sub_epi32(X2[0], GET0));
            X2[1] = _mm_add_epi32(temp[1], _mm_sub_epi32(X2[1], GET1));
            X2[2] = _mm_add_epi32(temp[2], _mm_sub_epi32(X2[2], GET2));
            X2[3] = _mm_add_epi32(temp[3], _mm_sub_epi32(X2[3], GET3));
            // c[x].r = (uint8_t) ((X2.r * X2div + (1 << 23)) >> 24);
            temp[0] = _mm_srli_epi32(_mm_add_epi32(_mm_mullo_epi32(X2[0], X2div), b23), 24);
            temp[1] = _mm_srli_epi32(_mm_add_epi32(_mm_mullo_epi32(X2[1], X2div), b23), 24);
            temp[2] = _mm_srli_epi32(_mm_add_epi32(_mm_mullo_epi32(X2[2], X2div), b23), 24);
            temp[3] = _mm_srli_epi32(_mm_add_epi32(_mm_mullo_epi32(X2[3], X2div), b23), 24);
            STORETEMP(c[x]);
            // X3.r += c[x].r;
            X3[0] = _mm_add_epi32(X3[0], temp[0]);
            X3[1] = _mm_add_epi32(X3[1], temp[1]);
            X3[2] = _mm_add_epi32(X3[2], temp[2]);
            X3[3] = _mm_add_epi32(X3[3], temp[3]);
        }

        b[-1 & r_mask] = b[0];
        for (size_t x = 0; x <= r; x += 1) {
            c[(x - r - 1) & r_mask] = c[0];
        }
        for (size_t x = 0; x < Simg->ysize - r3; x += 1) {
            // pixel32 last_b, last_c;
            // X1.r += sdata[x + r3].r - sdata[x + r - 1].r;
            LOADSDATA(x + r - 1);
            X1[0] = _mm_sub_epi32(X1[0], GET0);
            X1[1] = _mm_sub_epi32(X1[1], GET1);
            X1[2] = _mm_sub_epi32(X1[2], GET2);
            X1[3] = _mm_sub_epi32(X1[3], GET3);
            LOADSDATA(x + r3);
            X1[0] = _mm_add_epi32(X1[0], GET0);
            X1[1] = _mm_add_epi32(X1[1], GET1);
            X1[2] = _mm_add_epi32(X1[2], GET2);
            X1[3] = _mm_add_epi32(X1[3], GET3);
            // last_b.r = b[(x + r2) & r_mask].r = (uint8_t) ((X1.r * X1div + (1 << 23)) >> 24);
            temp[0] = _mm_srli_epi32(_mm_add_epi32(_mm_mullo_epi32(X1[0], X1div), b23), 24);
            temp[1] = _mm_srli_epi32(_mm_add_epi32(_mm_mullo_epi32(X1[1], X1div), b23), 24);
            temp[2] = _mm_srli_epi32(_mm_add_epi32(_mm_mullo_epi32(X1[2], X1div), b23), 24);
            temp[3] = _mm_srli_epi32(_mm_add_epi32(_mm_mullo_epi32(X1[3], X1div), b23), 24);
            STORETEMP(b[(x + r2) & r_mask]);
            // X2.r += last_b.r - b[(x - 1) & r_mask].r;
            LOAD(b[(x - 1) & r_mask]);
            X2[0] = _mm_add_epi32(temp[0], _mm_sub_epi32(X2[0], GET0));
            X2[1] = _mm_add_epi32(temp[1], _mm_sub_epi32(X2[1], GET1));
            X2[2] = _mm_add_epi32(temp[2], _mm_sub_epi32(X2[2], GET2));
            X2[3] = _mm_add_epi32(temp[3], _mm_sub_epi32(X2[3], GET3));
            // last_c.r = c[(x + r) & r_mask].r = (uint8_t) ((X2.r * X2div + (1 << 23)) >> 24);
            temp[0] = _mm_srli_epi32(_mm_add_epi32(_mm_mullo_epi32(X2[0], X2div), b23), 24);
            temp[1] = _mm_srli_epi32(_mm_add_epi32(_mm_mullo_epi32(X2[1], X2div), b23), 24);
            temp[2] = _mm_srli_epi32(_mm_add_epi32(_mm_mullo_epi32(X2[2], X2div), b23), 24);
            temp[3] = _mm_srli_epi32(_mm_add_epi32(_mm_mullo_epi32(X2[3], X2div), b23), 24);
            STORETEMP(c[(x + r) & r_mask]);
            // X3.r += last_c.r - c[(x - r - 1) & r_mask].r;
            LOAD(c[(x - r - 1) & r_mask]);
            X3[0] = _mm_add_epi32(temp[0], _mm_sub_epi32(X3[0], GET0));
            X3[1] = _mm_add_epi32(temp[1], _mm_sub_epi32(X3[1], GET1));
            X3[2] = _mm_add_epi32(temp[2], _mm_sub_epi32(X3[2], GET2));
            X3[3] = _mm_add_epi32(temp[3], _mm_sub_epi32(X3[3], GET3));

            // *rdata.r = (uint8_t) ((X3.r * X3div + (1 << 23)) >> 24);
            temp[0] = _mm_srli_epi32(_mm_add_epi32(_mm_mullo_epi32(X3[0], X3div), b23), 24);
            temp[1] = _mm_srli_epi32(_mm_add_epi32(_mm_mullo_epi32(X3[1], X3div), b23), 24);
            temp[2] = _mm_srli_epi32(_mm_add_epi32(_mm_mullo_epi32(X3[2], X3div), b23), 24);
            temp[3] = _mm_srli_epi32(_mm_add_epi32(_mm_mullo_epi32(X3[3], X3div), b23), 24);
            STORETEMP(line[x & 0x3]);

            if ((x & 0x3) == 0x3) {
                // Transpose
                __m128 tmp3, tmp2, tmp1, tmp0;
                tmp0 = _mm_unpacklo_ps((__m128) line[0], (__m128)line[1]);
                tmp2 = _mm_unpacklo_ps((__m128) line[2], (__m128)line[3]);
                tmp1 = _mm_unpackhi_ps((__m128) line[0], (__m128)line[1]);
                tmp3 = _mm_unpackhi_ps((__m128) line[2], (__m128)line[3]);
                line[0] = (__m128i) _mm_movelh_ps(tmp0, tmp2);
                line[1] = (__m128i) _mm_movehl_ps(tmp2, tmp0);
                line[2] = (__m128i) _mm_movelh_ps(tmp1, tmp3);
                line[3] = (__m128i) _mm_movehl_ps(tmp3, tmp1);

                _mm_storeu_si128((__m128i*) &rdata[(x & ~0x3) + 0 * Rimg->xsize], line[0]);
                _mm_storeu_si128((__m128i*) &rdata[(x & ~0x3) + 1 * Rimg->xsize], line[1]);
                _mm_storeu_si128((__m128i*) &rdata[(x & ~0x3) + 2 * Rimg->xsize], line[2]);
                _mm_storeu_si128((__m128i*) &rdata[(x & ~0x3) + 3 * Rimg->xsize], line[3]);
            }
        }

        for (size_t x = Simg->ysize - r3; x < Simg->ysize - r2; x += 1) {
            // pixel32 last_b, last_c;
            // X1.r += sdata[lastx].r - sdata[x + r - 1].r;
            LOADSDATA(x + r - 1);
            X1[0] = _mm_sub_epi32(X1[0], GET0);
            X1[1] = _mm_sub_epi32(X1[1], GET1);
            X1[2] = _mm_sub_epi32(X1[2], GET2);
            X1[3] = _mm_sub_epi32(X1[3], GET3);
            LOADSDATA(lastx);
            X1[0] = _mm_add_epi32(X1[0], GET0);
            X1[1] = _mm_add_epi32(X1[1], GET1);
            X1[2] = _mm_add_epi32(X1[2], GET2);
            X1[3] = _mm_add_epi32(X1[3], GET3);
            // last_b.r = b[(x + r2) & r_mask].r = (uint8_t) ((X1.r * X1div + (1 << 23)) >> 24);
            temp[0] = _mm_srli_epi32(_mm_add_epi32(_mm_mullo_epi32(X1[0], X1div), b23), 24);
            temp[1] = _mm_srli_epi32(_mm_add_epi32(_mm_mullo_epi32(X1[1], X1div), b23), 24);
            temp[2] = _mm_srli_epi32(_mm_add_epi32(_mm_mullo_epi32(X1[2], X1div), b23), 24);
            temp[3] = _mm_srli_epi32(_mm_add_epi32(_mm_mullo_epi32(X1[3], X1div), b23), 24);
            STORETEMP(b[(x + r2) & r_mask]);
            // X2.r += last_b.r - b[(x - 1) & r_mask].r;
            LOAD(b[(x - 1) & r_mask]);
            X2[0] = _mm_add_epi32(temp[0], _mm_sub_epi32(X2[0], GET0));
            X2[1] = _mm_add_epi32(temp[1], _mm_sub_epi32(X2[1], GET1));
            X2[2] = _mm_add_epi32(temp[2], _mm_sub_epi32(X2[2], GET2));
            X2[3] = _mm_add_epi32(temp[3], _mm_sub_epi32(X2[3], GET3));
            // last_c.r = c[(x + r) & r_mask].r = (uint8_t) ((X2.r * X2div + (1 << 23)) >> 24);
            temp[0] = _mm_srli_epi32(_mm_add_epi32(_mm_mullo_epi32(X2[0], X2div), b23), 24);
            temp[1] = _mm_srli_epi32(_mm_add_epi32(_mm_mullo_epi32(X2[1], X2div), b23), 24);
            temp[2] = _mm_srli_epi32(_mm_add_epi32(_mm_mullo_epi32(X2[2], X2div), b23), 24);
            temp[3] = _mm_srli_epi32(_mm_add_epi32(_mm_mullo_epi32(X2[3], X2div), b23), 24);
            STORETEMP(c[(x + r) & r_mask]);
            // X3.r += last_c.r - c[(x - r - 1) & r_mask].r;
            LOAD(c[(x - r - 1) & r_mask]);
            X3[0] = _mm_add_epi32(temp[0], _mm_sub_epi32(X3[0], GET0));
            X3[1] = _mm_add_epi32(temp[1], _mm_sub_epi32(X3[1], GET1));
            X3[2] = _mm_add_epi32(temp[2], _mm_sub_epi32(X3[2], GET2));
            X3[3] = _mm_add_epi32(temp[3], _mm_sub_epi32(X3[3], GET3));

            // *rdata.r = (uint8_t) ((X3.r * X3div + (1 << 23)) >> 24);
            temp[0] = _mm_srli_epi32(_mm_add_epi32(_mm_mullo_epi32(X3[0], X3div), b23), 24);
            temp[1] = _mm_srli_epi32(_mm_add_epi32(_mm_mullo_epi32(X3[1], X3div), b23), 24);
            temp[2] = _mm_srli_epi32(_mm_add_epi32(_mm_mullo_epi32(X3[2], X3div), b23), 24);
            temp[3] = _mm_srli_epi32(_mm_add_epi32(_mm_mullo_epi32(X3[3], X3div), b23), 24);
            STORETEMP(line[x & 0x3]);

            if ((x & 0x3) == 0x3) {
                // Transpose
                __m128 tmp3, tmp2, tmp1, tmp0;
                tmp0 = _mm_unpacklo_ps((__m128) line[0], (__m128)line[1]);
                tmp2 = _mm_unpacklo_ps((__m128) line[2], (__m128)line[3]);
                tmp1 = _mm_unpackhi_ps((__m128) line[0], (__m128)line[1]);
                tmp3 = _mm_unpackhi_ps((__m128) line[2], (__m128)line[3]);
                line[0] = (__m128i) _mm_movelh_ps(tmp0, tmp2);
                line[1] = (__m128i) _mm_movehl_ps(tmp2, tmp0);
                line[2] = (__m128i) _mm_movelh_ps(tmp1, tmp3);
                line[3] = (__m128i) _mm_movehl_ps(tmp3, tmp1);

                _mm_storeu_si128((__m128i*) &rdata[(x & ~0x3) + 0 * Rimg->xsize], line[0]);
                _mm_storeu_si128((__m128i*) &rdata[(x & ~0x3) + 1 * Rimg->xsize], line[1]);
                _mm_storeu_si128((__m128i*) &rdata[(x & ~0x3) + 2 * Rimg->xsize], line[2]);
                _mm_storeu_si128((__m128i*) &rdata[(x & ~0x3) + 3 * Rimg->xsize], line[3]);
            }
        }

        for (size_t x = Simg->ysize - r2; x < Simg->ysize - r; x += 1) {
            // pixel32 last_c;
            // X2.r += b[lastx & r_mask].r - b[(x - 1) & r_mask].r;
            LOAD(b[(x - 1) & r_mask]);
            X2[0] = _mm_sub_epi32(X2[0], GET0);
            X2[1] = _mm_sub_epi32(X2[1], GET1);
            X2[2] = _mm_sub_epi32(X2[2], GET2);
            X2[3] = _mm_sub_epi32(X2[3], GET3);
            LOAD(b[lastx & r_mask]);
            X2[0] = _mm_add_epi32(X2[0], GET0);
            X2[1] = _mm_add_epi32(X2[1], GET1);
            X2[2] = _mm_add_epi32(X2[2], GET2);
            X2[3] = _mm_add_epi32(X2[3], GET3);
            // last_c.r = c[(x + r) & r_mask].r = (uint8_t) ((X2.r * X2div + (1 << 23)) >> 24);
            temp[0] = _mm_srli_epi32(_mm_add_epi32(_mm_mullo_epi32(X2[0], X2div), b23), 24);
            temp[1] = _mm_srli_epi32(_mm_add_epi32(_mm_mullo_epi32(X2[1], X2div), b23), 24);
            temp[2] = _mm_srli_epi32(_mm_add_epi32(_mm_mullo_epi32(X2[2], X2div), b23), 24);
            temp[3] = _mm_srli_epi32(_mm_add_epi32(_mm_mullo_epi32(X2[3], X2div), b23), 24);
            STORETEMP(c[(x + r) & r_mask]);
            // X3.r += last_c.r - c[(x - r - 1) & r_mask].r;
            LOAD(c[(x - r - 1) & r_mask]);
            X3[0] = _mm_add_epi32(temp[0], _mm_sub_epi32(X3[0], GET0));
            X3[1] = _mm_add_epi32(temp[1], _mm_sub_epi32(X3[1], GET1));
            X3[2] = _mm_add_epi32(temp[2], _mm_sub_epi32(X3[2], GET2));
            X3[3] = _mm_add_epi32(temp[3], _mm_sub_epi32(X3[3], GET3));

            // *rdata.r = (uint8_t) ((X3.r * X3div + (1 << 23)) >> 24);
            temp[0] = _mm_srli_epi32(_mm_add_epi32(_mm_mullo_epi32(X3[0], X3div), b23), 24);
            temp[1] = _mm_srli_epi32(_mm_add_epi32(_mm_mullo_epi32(X3[1], X3div), b23), 24);
            temp[2] = _mm_srli_epi32(_mm_add_epi32(_mm_mullo_epi32(X3[2], X3div), b23), 24);
            temp[3] = _mm_srli_epi32(_mm_add_epi32(_mm_mullo_epi32(X3[3], X3div), b23), 24);
            STORETEMP(line[x & 0x3]);

            if ((x & 0x3) == 0x3) {
                // Transpose
                __m128 tmp3, tmp2, tmp1, tmp0;
                tmp0 = _mm_unpacklo_ps((__m128) line[0], (__m128)line[1]);
                tmp2 = _mm_unpacklo_ps((__m128) line[2], (__m128)line[3]);
                tmp1 = _mm_unpackhi_ps((__m128) line[0], (__m128)line[1]);
                tmp3 = _mm_unpackhi_ps((__m128) line[2], (__m128)line[3]);
                line[0] = (__m128i) _mm_movelh_ps(tmp0, tmp2);
                line[1] = (__m128i) _mm_movehl_ps(tmp2, tmp0);
                line[2] = (__m128i) _mm_movelh_ps(tmp1, tmp3);
                line[3] = (__m128i) _mm_movehl_ps(tmp3, tmp1);

                _mm_storeu_si128((__m128i*) &rdata[(x & ~0x3) + 0 * Rimg->xsize], line[0]);
                _mm_storeu_si128((__m128i*) &rdata[(x & ~0x3) + 1 * Rimg->xsize], line[1]);
                _mm_storeu_si128((__m128i*) &rdata[(x & ~0x3) + 2 * Rimg->xsize], line[2]);
                _mm_storeu_si128((__m128i*) &rdata[(x & ~0x3) + 3 * Rimg->xsize], line[3]);
            }
        }

        for (size_t x = Simg->ysize - r; x < Simg->ysize; x += 1) {
            // X3.r += c[lastx & r_mask].r - c[(x - r - 1) & r_mask].r;
            LOAD(c[(x - r - 1) & r_mask]);
            X3[0] = _mm_sub_epi32(X3[0], GET0);
            X3[1] = _mm_sub_epi32(X3[1], GET1);
            X3[2] = _mm_sub_epi32(X3[2], GET2);
            X3[3] = _mm_sub_epi32(X3[3], GET3);
            LOAD(c[lastx & r_mask]);
            X3[0] = _mm_add_epi32(X3[0], GET0);
            X3[1] = _mm_add_epi32(X3[1], GET1);
            X3[2] = _mm_add_epi32(X3[2], GET2);
            X3[3] = _mm_add_epi32(X3[3], GET3);

            // *rdata.r = (uint8_t) ((X3.r * X3div + (1 << 23)) >> 24);
            temp[0] = _mm_srli_epi32(_mm_add_epi32(_mm_mullo_epi32(X3[0], X3div), b23), 24);
            temp[1] = _mm_srli_epi32(_mm_add_epi32(_mm_mullo_epi32(X3[1], X3div), b23), 24);
            temp[2] = _mm_srli_epi32(_mm_add_epi32(_mm_mullo_epi32(X3[2], X3div), b23), 24);
            temp[3] = _mm_srli_epi32(_mm_add_epi32(_mm_mullo_epi32(X3[3], X3div), b23), 24);
            STORETEMP(line[x & 0x3]);

            if ((x & 0x3) == 0x3) {
                // Transpose
                __m128 tmp3, tmp2, tmp1, tmp0;
                tmp0 = _mm_unpacklo_ps((__m128) line[0], (__m128)line[1]);
                tmp2 = _mm_unpacklo_ps((__m128) line[2], (__m128)line[3]);
                tmp1 = _mm_unpackhi_ps((__m128) line[0], (__m128)line[1]);
                tmp3 = _mm_unpackhi_ps((__m128) line[2], (__m128)line[3]);
                line[0] = (__m128i) _mm_movelh_ps(tmp0, tmp2);
                line[1] = (__m128i) _mm_movehl_ps(tmp2, tmp0);
                line[2] = (__m128i) _mm_movelh_ps(tmp1, tmp3);
                line[3] = (__m128i) _mm_movehl_ps(tmp3, tmp1);

                _mm_storeu_si128((__m128i*) &rdata[(x & ~0x3) + 0 * Rimg->xsize], line[0]);
                _mm_storeu_si128((__m128i*) &rdata[(x & ~0x3) + 1 * Rimg->xsize], line[1]);
                _mm_storeu_si128((__m128i*) &rdata[(x & ~0x3) + 2 * Rimg->xsize], line[2]);
                _mm_storeu_si128((__m128i*) &rdata[(x & ~0x3) + 3 * Rimg->xsize], line[3]);
            }
        }
    }

    #undef LOADSDATA
    #undef LOAD
    #undef STORETEMP
    #undef GET0
    #undef GET1
    #undef GET2
    #undef GET3
}


static void
opTriBoxBlur_horz_smallr(
    image32* restrict Rimg,
    const image32* restrict Simg, float floatR)
{
    // Max floatR for this implementation is 64 (excluded).
    uint32_t r = floatR + 1;
    uint32_t d = r * 2 + 1;
    uint32_t r2 = r * 2, r3 = r * 3;
    // We need two extra slots for writing even if E1div = 0
    uint32_t r_mask = all_bits_mask(d);
    // Each accumulator (Xn) consists of:
    // * 8 bits — source data
    // * from 1 to 7 bits - for accumulating (max d = 127)
    // * not less than 8 bits remains for integer division
    uint16_t _XEdiv = (1 << 15) / (floatR * 2 + 1) + 0.5;
    __m128i XEdiv = _mm_unpacklo_epi16(
        _mm_set1_epi16(_XEdiv),
        _mm_set1_epi16((1 << 15) > (d - 2) * _XEdiv ?
            ((1 << 15) - (d - 2) * _XEdiv) / 2.0 + 0.5 : 0));
    __m128i X1[2];
    __m128i X2[2];
    __m128i X3[2];
    __m128i temp;
    __m128i line[4];
    __m128i b0[r_mask + 1], b1[r_mask + 1];
    __m128i c0[r_mask + 1], c1[r_mask + 1];
    size_t lastx = Simg->ysize - 1;

    assert(floatR < 64);
    assert(Simg->ysize >= r3);

    #define LOADSDATA(x) _mm_cvtepu8_epi16(*(__m128i*) &sdata0[(x) * Simg->xsize])

    #define LINE(X, a, b) _mm_packus_epi32( \
        _mm_srli_epi32(_mm_add_epi32(_mm_madd_epi16(_mm_unpacklo_epi16( \
                X, _mm_add_epi16(a, b)), XEdiv), \
            _mm_set1_epi32(1<<14)), 15), \
        _mm_srli_epi32(_mm_add_epi32(_mm_madd_epi16(_mm_unpackhi_epi16( \
                X, _mm_add_epi16(a, b)), XEdiv), \
            _mm_set1_epi32(1<<14)), 15))

    for (size_t y = 0; y < Simg->xsize; y += 2) {
        if (y > Simg->xsize - 2) {
            y = Simg->xsize - 2;
        }
        pixel32* sdata0 = (pixel32*) Simg->data + y;
        uint32_t* rdata0 = (uint32_t*) Rimg->data + Rimg->xsize * y;
        uint32_t* rdata1 = (uint32_t*) Rimg->data + Rimg->xsize * (y + 1);
        
        // X1.r = sdata[0].r * r;
        X1[0] = _mm_mullo_epi16(LOADSDATA(0), _mm_set1_epi16(r));
        for (size_t x = 1; x < r; x += 1) {
            // X1.r += sdata[x].r;
            X1[0] = _mm_add_epi16(X1[0], LOADSDATA(x));
        }

        // b[0].r = (uint8_t) ((X1.r * X1div + (sdata[0].r + sdata[r].r) * E1div + (1<<15)) >> 16);
        b0[0] = LINE(X1[0], LOADSDATA(0), LOADSDATA(r));
        // X2.r = b[0].r * (r - 1);
        X2[0] = _mm_mullo_epi16(b0[0], _mm_set1_epi16(r - 1));
        for (size_t x = 1; x <= r; x += 1) {
            // X1.r += sdata[x+r-1].r - sdata[0].r;
            X1[0] = _mm_add_epi16(_mm_sub_epi16(X1[0], LOADSDATA(0)), LOADSDATA(x+r-1));
            // b[x].r = (uint8_t) ((X1.r * X1div + (sdata[0].r + sdata[x+r].r) * E1div + (1<<15)) >> 16);
            b0[x] = LINE(X1[0], LOADSDATA(0), LOADSDATA(x+r));
            // X2.r += b[x-1].r;
            X2[0] = _mm_add_epi16(X2[0], b0[x-1]);
        }

        // c[0].r = (uint8_t) ((X2.r * X2div + (b[0].r + b[r].r) * E2div + (1<<15)) >> 16);
        c0[0] = LINE(X2[0], b0[0], b0[r]);
        // X3.r = c[0].r * r;
        X3[0] = _mm_mullo_epi16(c0[0], _mm_set1_epi16(r));
        for (size_t x = 1; x < r; x += 1) {
            // X1.r += sdata[x+r2-1].r - sdata[x].r;
            X1[0] = _mm_add_epi16(_mm_sub_epi16(X1[0], LOADSDATA(x)), LOADSDATA(x+r2-1));
            // b[x+r].r = (uint8_t) ((X1.r * X1div + (sdata[x].r + sdata[x+r2].r) * E1div + (1<<15)) >> 16);
            b0[x+r] = LINE(X1[0], LOADSDATA(x), LOADSDATA(x+r2));
            // X2.r += b[x+r-1].r - b[0].r;
            X2[0] = _mm_add_epi16(_mm_sub_epi16(X2[0], b0[0]), b0[x+r-1]);
            // c[x].r = (uint8_t) ((X2.r * X2div + (b[0].r + b[x+r].r) * E2div + (1<<15)) >> 16);
            c0[x] = LINE(X2[0], b0[0], b0[x+r]);
            // X3.r += c[x-1].r;
            X3[0] = _mm_add_epi16(X3[0], c0[x-1]);
        }

        b0[-1 & r_mask] = b0[0];
        for (size_t x = 0; x <= r; x += 1) {
            c0[(x-r-1) & r_mask] = c0[0];
        }
        for (size_t x = 0; x < Simg->ysize - r3; x += 1) {
            __m128i last_b0, last_c0;
            // X1.r += sdata[x+r3-1].r - sdata[x+r].r;
            X1[0] = _mm_add_epi16(_mm_sub_epi16(X1[0], LOADSDATA(x+r)), LOADSDATA(x+r3-1));
            // last_b.r = b[(x+r2) & r_mask].r = (uint8_t) ((X1.r * X1div + (sdata[x+r].r + sdata[x+r3].r) * E1div + (1<<15)) >> 16);
            last_b0 = b0[(x+r2) & r_mask] = LINE(X1[0], LOADSDATA(x+r), LOADSDATA(x+r3));
            // X2.r += b[(x+r2-1) & r_mask].r - b[x & r_mask].r;
            X2[0] = _mm_add_epi16(_mm_sub_epi16(X2[0], b0[x & r_mask]), b0[(x+r2-1) & r_mask]);
            // last_c.r = c[(x+r) & r_mask].r = (uint8_t) ((X2.r * X2div + (b[x & r_mask].r + last_b.r) * E2div + (1<<15)) >> 16);
            last_c0 = c0[(x+r) & r_mask] = LINE(X2[0], b0[x & r_mask], last_b0);
            // X3.r += c[(x+r-1) & r_mask].r - c[(x-r) & r_mask].r;
            X3[0] = _mm_add_epi16(_mm_sub_epi16(X3[0], c0[(x-r) & r_mask]), c0[(x+r-1) & r_mask]);

            // *rdata = (uint8_t) ((X3.r * X3div + (c[(x-r) & r_mask].r + last_c.r) * E3div + (1<<15)) >> 16);
            temp = _mm_packus_epi16(LINE(X3[0], c0[(x-r) & r_mask], last_c0), _mm_setzero_si128());
            rdata0[x] = _mm_cvtsi128_si32(temp);
            rdata1[x] = _mm_cvtsi128_si32(_mm_srli_si128(temp, 4));
        }

        for (size_t x = Simg->ysize - r3; x < Simg->ysize - r2; x += 1) {
            __m128i last_b0, last_c0;
            // X1.r += sdata[lastx].r - sdata[x+r].r;
            X1[0] = _mm_add_epi16(_mm_sub_epi16(X1[0], LOADSDATA(x+r)), LOADSDATA(lastx));
            // last_b.r = b[(x+r2) & r_mask].r = (uint8_t) ((X1.r * X1div + (sdata[x+r].r + sdata[lastx].r) * E1div + (1<<15)) >> 16);
            last_b0 = b0[(x+r2) & r_mask] = LINE(X1[0], LOADSDATA(x+r), LOADSDATA(lastx));
            // X2.r += b[(x+r2-1) & r_mask].r - b[x & r_mask].r;
            X2[0] = _mm_add_epi16(_mm_sub_epi16(X2[0], b0[x & r_mask]), b0[(x+r2-1) & r_mask]);
            // last_c.r = c[(x+r) & r_mask].r = (uint8_t) ((X2.r * X2div + (b[x & r_mask].r + last_b.r) * E2div + (1<<15)) >> 16);
            last_c0 = c0[(x+r) & r_mask] = LINE(X2[0], b0[x & r_mask], last_b0);
            // X3.r += c[(x+r-1) & r_mask].r - c[(x-r) & r_mask].r;
            X3[0] = _mm_add_epi16(_mm_sub_epi16(X3[0], c0[(x-r) & r_mask]), c0[(x+r-1) & r_mask]);

            // *rdata = (uint8_t) ((X3.r * X3div + (c[(x-r) & r_mask].r + last_c.r) * E3div + (1<<15)) >> 16);
            temp = _mm_packus_epi16(LINE(X3[0], c0[(x-r) & r_mask], last_c0), _mm_setzero_si128());
            rdata0[x] = _mm_cvtsi128_si32(temp);
            rdata1[x] = _mm_cvtsi128_si32(_mm_srli_si128(temp, 4));
        }

        for (size_t x = Simg->ysize - r2; x < Simg->ysize - r; x += 1) {
            __m128i last_c0;
            // X2.r += b[lastx & r_mask].r - b[x & r_mask].r;
            X2[0] = _mm_add_epi16(_mm_sub_epi16(X2[0], b0[x & r_mask]), b0[lastx & r_mask]);
            // last_c.r = c[(x+r) & r_mask].r = (uint8_t) ((X2.r * X2div + (b[x & r_mask].r + b[lastx & r_mask].r) * E2div + (1<<15)) >> 16);
            last_c0 = c0[(x+r) & r_mask] = LINE(X2[0], b0[x & r_mask], b0[lastx & r_mask]);
            // X3.r += c[(x+r-1) & r_mask].r - c[(x-r) & r_mask].r;
            X3[0] = _mm_add_epi16(_mm_sub_epi16(X3[0], c0[(x-r) & r_mask]), c0[(x+r-1) & r_mask]);

            // *rdata = (uint8_t) ((X3.r * X3div + (c[(x-r) & r_mask].r + last_c.r) * E3div + (1<<15)) >> 16);
            temp = _mm_packus_epi16(LINE(X3[0], c0[(x-r) & r_mask], last_c0), _mm_setzero_si128());
            rdata0[x] = _mm_cvtsi128_si32(temp);
            rdata1[x] = _mm_cvtsi128_si32(_mm_srli_si128(temp, 4));
        }

        for (size_t x = Simg->ysize - r; x < Simg->ysize; x += 1) {
            // X3.r += c[lastx & r_mask].r - c[(x-r) & r_mask].r;
            X3[0] = _mm_add_epi16(_mm_sub_epi16(X3[0], c0[(x-r) & r_mask]), c0[lastx & r_mask]);

            // *rdata = (uint8_t) ((X3.r * X3div + (c[(x-r) & r_mask].r + c[lastx & r_mask].r) * E3div + (1<<15)) >> 16);
            temp = _mm_packus_epi16(LINE(X3[0], c0[(x-r) & r_mask], c0[lastx & r_mask]), _mm_setzero_si128());
            rdata0[x] = _mm_cvtsi128_si32(temp);
            rdata1[x] = _mm_cvtsi128_si32(_mm_srli_si128(temp, 4));
        }
    }
}


extern void
opTriBoxBlur_premul(
    image32* restrict Rimage,
    image32* restrict INTimage,
    const image32* restrict Simage, float r)
{
    if (r < 64) {
        opTriBoxBlur_horz_smallr(INTimage, Simage, r);
        opTriBoxBlur_horz_smallr(Rimage, INTimage, r);
    } else {
        opTriBoxBlur_horz_larger(INTimage, Simage, r + 0.5);
        opTriBoxBlur_horz_larger(Rimage, INTimage, r + 0.5);
    }
}
