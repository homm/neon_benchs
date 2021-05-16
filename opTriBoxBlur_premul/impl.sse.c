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
    __m128i storemask = _mm_set_epi8(0,0,0,0,0,0,0,0,0,0,0,0,15,11,7,3);
    __m128i X1[2];
    __m128i X2[2];
    __m128i X3[2];
    __m128i temp0, temp1;
    __m128i b[r_mask + 1];
    __m128i c[r_mask + 1];
    __m128i zero = _mm_setzero_si128();
    size_t lastx = Simg->xsize - 1;

    assert(r < (1 << 15));
    assert(Simg->xsize >= r3);

    for (size_t y = 0; y < Simg->ysize - 1; y += 2) {
        pixel32* sdata0 = (pixel32*) Simg->data + Simg->xsize * y;
        pixel32* sdata1 = (pixel32*) Simg->data + Simg->xsize * (y + 1);
        pixel32* rdata = (pixel32*) Rimg->data + y;
        
        // X1.r = sdata0[0].r * (r + 1);
        X1[0] = _mm_mullo_epi32(mm_cvtepu8_epi32(&sdata0[0]), _mm_set1_epi32(r + 1));
        X1[1] = _mm_mullo_epi32(mm_cvtepu8_epi32(&sdata1[0]), _mm_set1_epi32(r + 1));
        for (size_t x = 1; x <= r; x += 1) {
            // X1.r += sdata0[x].r;
            X1[0] = _mm_add_epi32(X1[0], mm_cvtepu8_epi32(&sdata0[x]));
            X1[1] = _mm_add_epi32(X1[1], mm_cvtepu8_epi32(&sdata1[x]));
        }

        // b[0].r = (uint8_t) ((X1.r * X1div + (1 << 23)) >> 24);
        temp0 = _mm_srli_epi32(_mm_add_epi32(_mm_mullo_epi32(X1[0], X1div), b23), 24);
        temp1 = _mm_srli_epi32(_mm_add_epi32(_mm_mullo_epi32(X1[1], X1div), b23), 24);
        b[0] = _mm_packs_epi32(temp0, temp1);
        // X2.r = b[0].r * (r + 1);
        X2[0] = _mm_mullo_epi32(temp0, _mm_set1_epi32(r + 1));
        X2[1] = _mm_mullo_epi32(temp1, _mm_set1_epi32(r + 1));
        for (size_t x = 1; x <= r; x += 1) {
            // X1.r += sdata0[x + r].r - sdata0[0].r;
            X1[0] = _mm_add_epi32(mm_cvtepu8_epi32(&sdata0[x + r]),
                _mm_sub_epi32(X1[0], mm_cvtepu8_epi32(&sdata0[0])));
            X1[1] = _mm_add_epi32(mm_cvtepu8_epi32(&sdata1[x + r]),
                _mm_sub_epi32(X1[1], mm_cvtepu8_epi32(&sdata1[0])));
            // b[x].r = (uint8_t) ((X1.r * X1div + (1 << 23)) >> 24);
            temp0 = _mm_srli_epi32(_mm_add_epi32(_mm_mullo_epi32(X1[0], X1div), b23), 24);
            temp1 = _mm_srli_epi32(_mm_add_epi32(_mm_mullo_epi32(X1[1], X1div), b23), 24);
            b[x] = _mm_packs_epi32(temp0, temp1);
            // X2.r += b[x].r;
            X2[0] = _mm_add_epi32(X2[0], temp0);
            X2[1] = _mm_add_epi32(X2[1], temp1);
        }

        // c[0].r = (uint8_t) ((X2.r * X2div + (1 << 23)) >> 24);
        temp0 = _mm_srli_epi32(_mm_add_epi32(_mm_mullo_epi32(X2[0], X2div), b23), 24);
        temp1 = _mm_srli_epi32(_mm_add_epi32(_mm_mullo_epi32(X2[1], X2div), b23), 24);
        c[0] =  _mm_packs_epi32(temp0, temp1);
        // X3.r = c[0].r * (r + 2);
        X3[0] = _mm_mullo_epi32(temp0, _mm_set1_epi32(r + 2));
        X3[1] = _mm_mullo_epi32(temp1, _mm_set1_epi32(r + 2));
        for (size_t x = 1; x < r; x += 1) {
            // X1.r += sdata0[x + r2].r - sdata0[x - 1].r;
            X1[0] = _mm_add_epi32(mm_cvtepu8_epi32(&sdata0[x + r2]),
                _mm_sub_epi32(X1[0], mm_cvtepu8_epi32(&sdata0[x - 1])));
            X1[1] = _mm_add_epi32(mm_cvtepu8_epi32(&sdata1[x + r2]),
                _mm_sub_epi32(X1[1], mm_cvtepu8_epi32(&sdata1[x - 1])));
            // b[x + r].r = (uint8_t) ((X1.r * X1div + (1 << 23)) >> 24);
            temp0 = _mm_srli_epi32(_mm_add_epi32(_mm_mullo_epi32(X1[0], X1div), b23), 24);
            temp1 = _mm_srli_epi32(_mm_add_epi32(_mm_mullo_epi32(X1[1], X1div), b23), 24);
            b[x + r] = _mm_packs_epi32(temp0, temp1);
            // X2.r += b[x + r].r - b[0].r;
            X2[0] = _mm_add_epi32(temp0, _mm_sub_epi32(X2[0], _mm_unpacklo_epi16(b[0], zero)));
            X2[1] = _mm_add_epi32(temp1, _mm_sub_epi32(X2[1], _mm_unpackhi_epi16(b[0], zero)));
            // c[x].r = (uint8_t) ((X2.r * X2div + (1 << 23)) >> 24);
            temp0 = _mm_srli_epi32(_mm_add_epi32(_mm_mullo_epi32(X2[0], X2div), b23), 24);
            temp1 = _mm_srli_epi32(_mm_add_epi32(_mm_mullo_epi32(X2[1], X2div), b23), 24);
            c[x] =  _mm_packs_epi32(temp0, temp1);
            // X3.r += c[x].r;
            X3[0] = _mm_add_epi32(X3[0], temp0);
            X3[1] = _mm_add_epi32(X3[1], temp1);
        }

        b[-1 & r_mask] = b[0];
        for (size_t x = 0; x <= r; x += 1) {
            c[(x - r - 1) & r_mask] = c[0];
        }
        for (size_t x = 0; x < Simg->xsize - r3; x += 1) {
            // pixel32 last_b, last_c;
            // X1.r += sdata0[x + r3].r - sdata0[x + r - 1].r;
            X1[0] = _mm_add_epi32(mm_cvtepu8_epi32(&sdata0[x + r3]),
                _mm_sub_epi32(X1[0], mm_cvtepu8_epi32(&sdata0[x + r - 1])));
            X1[1] = _mm_add_epi32(mm_cvtepu8_epi32(&sdata1[x + r3]),
                _mm_sub_epi32(X1[1], mm_cvtepu8_epi32(&sdata1[x + r - 1])));
            // last_b.r = b[(x + r2) & r_mask].r = (uint8_t) ((X1.r * X1div + (1 << 23)) >> 24);
            temp0 = _mm_srli_epi32(_mm_add_epi32(_mm_mullo_epi32(X1[0], X1div), b23), 24);
            temp1 = _mm_srli_epi32(_mm_add_epi32(_mm_mullo_epi32(X1[1], X1div), b23), 24);
            b[(x + r2) & r_mask] = _mm_packus_epi32(temp0, temp1);
            // X2.r += last_b.r - b[(x - 1) & r_mask].r;
            X2[0] = _mm_add_epi32(temp0, _mm_sub_epi32(X2[0], _mm_unpacklo_epi16(b[(x - 1) & r_mask], zero)));
            X2[1] = _mm_add_epi32(temp1, _mm_sub_epi32(X2[1], _mm_unpackhi_epi16(b[(x - 1) & r_mask], zero)));
            // last_c.r = c[(x + r) & r_mask].r = (uint8_t) ((X2.r * X2div + (1 << 23)) >> 24);
            temp0 = _mm_srli_epi32(_mm_add_epi32(_mm_mullo_epi32(X2[0], X2div), b23), 24);
            temp1 = _mm_srli_epi32(_mm_add_epi32(_mm_mullo_epi32(X2[1], X2div), b23), 24);
            c[(x + r) & r_mask] = _mm_packus_epi32(temp0, temp1);
            // X3.r += last_c.r - c[(x - r - 1) & r_mask].r;
            X3[0] = _mm_add_epi32(temp0, _mm_sub_epi32(X3[0], _mm_unpacklo_epi16(c[(x - r - 1) & r_mask], zero)));
            X3[1] = _mm_add_epi32(temp1, _mm_sub_epi32(X3[1], _mm_unpackhi_epi16(c[(x - r - 1) & r_mask], zero)));

            // *rdata.r = (uint8_t) ((X3.r * X3div + (1 << 23)) >> 24);
            temp0 = _mm_srli_epi32(_mm_add_epi32(_mm_mullo_epi32(X3[0], X3div), b23), 24);
            temp1 = _mm_srli_epi32(_mm_add_epi32(_mm_mullo_epi32(X3[1], X3div), b23), 24);
            _mm_storel_epi64((__m128i*) rdata, _mm_packus_epi16(_mm_packus_epi32(temp0, temp1), zero));
            rdata += Rimg->xsize;
        }

        for (size_t x = Simg->xsize - r3; x < Simg->xsize - r2; x += 1) {
            // pixel32 last_b, last_c;
            // X1.r += sdata0[lastx].r - sdata0[x + r - 1].r;
            X1[0] = _mm_add_epi32(mm_cvtepu8_epi32(&sdata0[lastx]),
                _mm_sub_epi32(X1[0], mm_cvtepu8_epi32(&sdata0[x + r - 1])));
            X1[1] = _mm_add_epi32(mm_cvtepu8_epi32(&sdata1[lastx]),
                _mm_sub_epi32(X1[1], mm_cvtepu8_epi32(&sdata1[x + r - 1])));
            // last_b.r = b[(x + r2) & r_mask].r = (uint8_t) ((X1.r * X1div + (1 << 23)) >> 24);
            temp0 = _mm_srli_epi32(_mm_add_epi32(_mm_mullo_epi32(X1[0], X1div), b23), 24);
            temp1 = _mm_srli_epi32(_mm_add_epi32(_mm_mullo_epi32(X1[1], X1div), b23), 24);
            b[(x + r2) & r_mask] = _mm_packus_epi32(temp0, temp1);
            // X2.r += last_b.r - b[(x - 1) & r_mask].r;
            X2[0] = _mm_add_epi32(temp0, _mm_sub_epi32(X2[0], _mm_unpacklo_epi16(b[(x - 1) & r_mask], zero)));
            X2[1] = _mm_add_epi32(temp1, _mm_sub_epi32(X2[1], _mm_unpackhi_epi16(b[(x - 1) & r_mask], zero)));
            // last_c.r = c[(x + r) & r_mask].r = (uint8_t) ((X2.r * X2div + (1 << 23)) >> 24);
            temp0 = _mm_srli_epi32(_mm_add_epi32(_mm_mullo_epi32(X2[0], X2div), b23), 24);
            temp1 = _mm_srli_epi32(_mm_add_epi32(_mm_mullo_epi32(X2[1], X2div), b23), 24);
            c[(x + r) & r_mask] = _mm_packus_epi32(temp0, temp1);
            // X3.r += last_c.r - c[(x - r - 1) & r_mask].r;
            X3[0] = _mm_add_epi32(temp0, _mm_sub_epi32(X3[0], _mm_unpacklo_epi16(c[(x - r - 1) & r_mask], zero)));
            X3[1] = _mm_add_epi32(temp1, _mm_sub_epi32(X3[1], _mm_unpackhi_epi16(c[(x - r - 1) & r_mask], zero)));

            // *rdata.r = (uint8_t) ((X3.r * X3div + (1 << 23)) >> 24);
            temp0 = _mm_srli_epi32(_mm_add_epi32(_mm_mullo_epi32(X3[0], X3div), b23), 24);
            temp1 = _mm_srli_epi32(_mm_add_epi32(_mm_mullo_epi32(X3[1], X3div), b23), 24);
            _mm_storel_epi64((__m128i*) rdata, _mm_packus_epi16(_mm_packus_epi32(temp0, temp1), zero));
            rdata += Rimg->xsize;
        }

        for (size_t x = Simg->xsize - r2; x < Simg->xsize - r; x += 1) {
            // pixel32 last_c;
            // X2.r += b[lastx & r_mask].r - b[(x - 1) & r_mask].r;
            X2[0] = _mm_add_epi32(_mm_unpacklo_epi16(b[lastx & r_mask], zero),
                _mm_sub_epi32(X2[0], _mm_unpacklo_epi16(b[(x - 1) & r_mask], zero)));
            X2[1] = _mm_add_epi32(_mm_unpackhi_epi16(b[lastx & r_mask], zero),
                _mm_sub_epi32(X2[1], _mm_unpackhi_epi16(b[(x - 1) & r_mask], zero)));
            // last_c.r = c[(x + r) & r_mask].r = (uint8_t) ((X2.r * X2div + (1 << 23)) >> 24);
            temp0 = _mm_srli_epi32(_mm_add_epi32(_mm_mullo_epi32(X2[0], X2div), b23), 24);
            temp1 = _mm_srli_epi32(_mm_add_epi32(_mm_mullo_epi32(X2[1], X2div), b23), 24);
            c[(x + r) & r_mask] = _mm_packus_epi32(temp0, temp1);
            // X3.r += last_c.r - c[(x - r - 1) & r_mask].r;
            X3[0] = _mm_add_epi32(temp0, _mm_sub_epi32(X3[0], _mm_unpacklo_epi16(c[(x - r - 1) & r_mask], zero)));
            X3[1] = _mm_add_epi32(temp1, _mm_sub_epi32(X3[1], _mm_unpackhi_epi16(c[(x - r - 1) & r_mask], zero)));

            // *rdata.r = (uint8_t) ((X3.r * X3div + (1 << 23)) >> 24);
            temp0 = _mm_srli_epi32(_mm_add_epi32(_mm_mullo_epi32(X3[0], X3div), b23), 24);
            temp1 = _mm_srli_epi32(_mm_add_epi32(_mm_mullo_epi32(X3[1], X3div), b23), 24);
            _mm_storel_epi64((__m128i*) rdata, _mm_packus_epi16(_mm_packus_epi32(temp0, temp1), zero));
            rdata += Rimg->xsize;
        }

        for (size_t x = Simg->xsize - r; x < Simg->xsize; x += 1) {
            // X3.r += c[lastx & r_mask].r - c[(x - r - 1) & r_mask].r;
            X3[0] = _mm_add_epi32(_mm_unpacklo_epi16(c[lastx & r_mask], zero),
                _mm_sub_epi32(X3[0], _mm_unpacklo_epi16(c[(x - r - 1) & r_mask], zero)));
            X3[1] = _mm_add_epi32(_mm_unpackhi_epi16(c[lastx & r_mask], zero),
                _mm_sub_epi32(X3[1], _mm_unpackhi_epi16(c[(x - r - 1) & r_mask], zero)));

            // *rdata.r = (uint8_t) ((X3.r * X3div + (1 << 23)) >> 24);
            temp0 = _mm_srli_epi32(_mm_add_epi32(_mm_mullo_epi32(X3[0], X3div), b23), 24);
            temp1 = _mm_srli_epi32(_mm_add_epi32(_mm_mullo_epi32(X3[1], X3div), b23), 24);
            _mm_storel_epi64((__m128i*) rdata, _mm_packus_epi16(_mm_packus_epi32(temp0, temp1), zero));
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
    uint32_t r = floatR + 1;
    uint32_t d = r * 2 + 1;
    uint32_t r2 = r * 2, r3 = r * 3;
    // We need two extra slots for writing even if E1div = 0
    uint32_t r_mask = all_bits_mask(d);
    // Each accumulator (Xn) consists of:
    // * 8 bits — source data
    // * from 1 to 8 bits - for accumulating (max d = 255)
    // * not less than 8 bits remains for integer division
    __m128i X1div = _mm_set1_epi16((1 << 16) / (floatR * 2 + 1) + 0.5);
    __m128i X2div = X1div, X3div = X1div;
    uint16_t _X1div = (1 << 16) / (floatR * 2 + 1) + 0.5;
    __m128i E1div = _mm_set1_epi16((1 << 16) > (d - 2) * _X1div ?
        ((1 << 16) - (d - 2) * _X1div) / 2.0 + 0.5 : 0);
    __m128i E2div = E1div, E3div = E1div;
    __m128i X1;
    __m128i X2;
    __m128i X3;
    __m128i b[r_mask + 1];
    __m128i c[r_mask + 1];
    size_t lastx = Simg->xsize - 1;

    assert(floatR < 128);
    assert(Simg->xsize >= r3);

    #define LOADSDATA(x) _mm_packus_epi32( \
        _mm_cvtepu8_epi32(*(__m128i*) &sdata0[x]), \
        _mm_cvtepu8_epi32(*(__m128i*) &sdata1[x]))

    for (size_t y = 0; y < Simg->ysize - 1; y += 2) {
        pixel32* sdata0 = (pixel32*) Simg->data + Simg->xsize * y;
        pixel32* sdata1 = (pixel32*) Simg->data + Simg->xsize * (y + 1);
        pixel32* rdata = (pixel32*) Rimg->data + y;
        
        // X1.r = sdata[0].r * r;
        X1 = _mm_mullo_epi16(LOADSDATA(0), _mm_set1_epi16(r));
        for (size_t x = 1; x < r; x += 1) {
            // X1.r += sdata[x].r;
            X1 = _mm_add_epi16(X1, LOADSDATA(x));
        }

        // b[0].r = (uint8_t) ((X1.r * X1div + (sdata[0].r + sdata[r].r) * E1div + (1<<15)) >> 16);
        b[0] = _mm_add_epi16(_mm_mulhi_epu16(X1, X1div),
            _mm_mulhi_epu16(_mm_add_epi16(LOADSDATA(0), LOADSDATA(r)), E1div));
        // X2.r = b[0].r * (r - 1);
        X2 = _mm_mullo_epi16(b[0], _mm_set1_epi16(r - 1));
        for (size_t x = 1; x <= r; x += 1) {
            // X1.r += sdata[x+r-1].r - sdata[0].r;
            X1 = _mm_add_epi16(_mm_sub_epi16(X1, LOADSDATA(0)), LOADSDATA(x+r-1));
            // b[x].r = (uint8_t) ((X1.r * X1div + (sdata[0].r + sdata[x+r].r) * E1div + (1<<15)) >> 16);
            b[x] = _mm_add_epi16(_mm_mulhi_epu16(X1, X1div),
                _mm_mulhi_epu16(_mm_add_epi16(LOADSDATA(0), LOADSDATA(x+r)), E1div));
            // X2.r += b[x-1].r;
            X2 = _mm_add_epi16(X2, b[x-1]);
        }

        // c[0].r = (uint8_t) ((X2.r * X2div + (b[0].r + b[r].r) * E2div + (1<<15)) >> 16);
        c[0] = _mm_add_epi16(_mm_mulhi_epu16(X2, X2div),
            _mm_mulhi_epu16(_mm_add_epi16(b[0], b[r]), E2div));
        // X3.r = c[0].r * r;
        X3 = _mm_mullo_epi16(c[0], _mm_set1_epi16(r));
        for (size_t x = 1; x < r; x += 1) {
            // X1.r += sdata[x+r2-1].r - sdata[x].r;
            X1 = _mm_add_epi16(_mm_sub_epi16(X1, LOADSDATA(x)), LOADSDATA(x+r2-1));
            // b[x+r].r = (uint8_t) ((X1.r * X1div + (sdata[x].r + sdata[x+r2].r) * E1div + (1<<15)) >> 16);
            b[x+r] = _mm_add_epi16(_mm_mulhi_epu16(X1, X1div),
                _mm_mulhi_epu16(_mm_add_epi16(LOADSDATA(x), LOADSDATA(x+r2)), E1div));
            // X2.r += b[x+r-1].r - b[0].r;
            X2 = _mm_add_epi16(_mm_sub_epi16(X2, b[0]), b[x+r-1]);
            // c[x].r = (uint8_t) ((X2.r * X2div + (b[0].r + b[x+r].r) * E2div + (1<<15)) >> 16);
            c[x] = _mm_add_epi16(_mm_mulhi_epu16(X2, X2div),
                _mm_mulhi_epu16(_mm_add_epi16(b[0], b[x+r]), E2div));
            // X3.r += c[x-1].r;
            X3 = _mm_add_epi16(X3, c[x-1]);
        }

        b[-1 & r_mask] = b[0];
        for (size_t x = 0; x <= r; x += 1) {
            c[(x-r-1) & r_mask] = c[0];
        }
        for (size_t x = 0; x < Simg->xsize - r3; x += 1) {
            __m128i last_b, last_c;
            // X1.r += sdata[x+r3-1].r - sdata[x+r].r;
            X1 = _mm_add_epi16(_mm_sub_epi16(X1, LOADSDATA(x+r)), LOADSDATA(x+r3-1));
            // last_b.r = b[(x+r2) & r_mask].r = (uint8_t) ((X1.r * X1div + (sdata[x+r].r + sdata[x+r3].r) * E1div + (1<<15)) >> 16);
            last_b = b[(x+r2) & r_mask] = _mm_add_epi16(_mm_mulhi_epu16(X1, X1div),
                _mm_mulhi_epu16(_mm_add_epi16(LOADSDATA(x+r), LOADSDATA(x+r3)), E1div));
            // X2.r += b[(x+r2-1) & r_mask].r - b[x & r_mask].r;
            X2 = _mm_add_epi16(_mm_sub_epi16(X2, b[x & r_mask]), b[(x+r2-1) & r_mask]);
            // last_c.r = c[(x+r) & r_mask].r = (uint8_t) ((X2.r * X2div + (b[x & r_mask].r + last_b.r) * E2div + (1<<15)) >> 16);
            last_c = c[(x+r) & r_mask] = _mm_add_epi16(_mm_mulhi_epu16(X2, X2div),
                _mm_mulhi_epu16(_mm_add_epi16(b[x & r_mask], last_b), E2div));
            // X3.r += c[(x+r-1) & r_mask].r - c[(x-r) & r_mask].r;
            X3 = _mm_add_epi16(_mm_sub_epi16(X3, c[(x-r) & r_mask]), c[(x+r-1) & r_mask]);

            // *rdata = (uint8_t) ((X3.r * X3div + (c[(x-r) & r_mask].r + last_c.r) * E3div + (1<<15)) >> 16);
            __m128i temp = _mm_add_epi16(_mm_mulhi_epu16(X3, X3div),
                _mm_mulhi_epu16(_mm_add_epi16(c[(x-r) & r_mask], last_c), E3div));
            _mm_storel_epi64((__m128i*) rdata, _mm_packus_epi16(temp, _mm_setzero_si128()));
            rdata += Rimg->xsize;
        }

        for (size_t x = Simg->xsize - r3; x < Simg->xsize - r2; x += 1) {
            __m128i last_b, last_c;
            // X1.r += sdata[lastx].r - sdata[x+r].r;
            X1 = _mm_add_epi16(_mm_sub_epi16(X1, LOADSDATA(x+r)), LOADSDATA(lastx));
            // last_b.r = b[(x+r2) & r_mask].r = (uint8_t) ((X1.r * X1div + (sdata[x+r].r + sdata[lastx].r) * E1div + (1<<15)) >> 16);
            last_b = b[(x+r2) & r_mask] = _mm_add_epi16(_mm_mulhi_epu16(X1, X1div),
                _mm_mulhi_epu16(_mm_add_epi16(LOADSDATA(x+r), LOADSDATA(lastx)), E1div));
            // X2.r += b[(x+r2-1) & r_mask].r - b[x & r_mask].r;
            X2 = _mm_add_epi16(_mm_sub_epi16(X2, b[x & r_mask]), b[(x+r2-1) & r_mask]);
            // last_c.r = c[(x+r) & r_mask].r = (uint8_t) ((X2.r * X2div + (b[x & r_mask].r + last_b.r) * E2div + (1<<15)) >> 16);
            last_c = c[(x+r) & r_mask] = _mm_add_epi16(_mm_mulhi_epu16(X2, X2div),
                _mm_mulhi_epu16(_mm_add_epi16(b[x & r_mask], last_b), E2div));
            // X3.r += c[(x+r-1) & r_mask].r - c[(x-r) & r_mask].r;
            X3 = _mm_add_epi16(_mm_sub_epi16(X3, c[(x-r) & r_mask]), c[(x+r-1) & r_mask]);

            // *rdata = (uint8_t) ((X3.r * X3div + (c[(x-r) & r_mask].r + last_c.r) * E3div + (1<<15)) >> 16);
            __m128i temp = _mm_add_epi16(_mm_mulhi_epu16(X3, X3div),
                _mm_mulhi_epu16(_mm_add_epi16(c[(x-r) & r_mask], last_c), E3div));
            _mm_storel_epi64((__m128i*) rdata, _mm_packus_epi16(temp, _mm_setzero_si128()));
            rdata += Rimg->xsize;
        }

        for (size_t x = Simg->xsize - r2; x < Simg->xsize - r; x += 1) {
            __m128i last_c;
            // X2.r += b[lastx & r_mask].r - b[x & r_mask].r;
            X2 = _mm_add_epi16(_mm_sub_epi16(X2, b[x & r_mask]), b[lastx & r_mask]);
            // last_c.r = c[(x+r) & r_mask].r = (uint8_t) ((X2.r * X2div + (b[x & r_mask].r + b[lastx & r_mask].r) * E2div + (1<<15)) >> 16);
            last_c = c[(x+r) & r_mask] = _mm_add_epi16(_mm_mulhi_epu16(X2, X2div),
                _mm_mulhi_epu16(_mm_add_epi16(b[x & r_mask], b[lastx & r_mask]), E2div));
            // X3.r += c[(x+r-1) & r_mask].r - c[(x-r) & r_mask].r;
            X3 = _mm_add_epi16(_mm_sub_epi16(X3, c[(x-r) & r_mask]), c[(x+r-1) & r_mask]);

            // *rdata = (uint8_t) ((X3.r * X3div + (c[(x-r) & r_mask].r + last_c.r) * E3div + (1<<15)) >> 16);
            __m128i temp = _mm_add_epi16(_mm_mulhi_epu16(X3, X3div),
                _mm_mulhi_epu16(_mm_add_epi16(c[(x-r) & r_mask], last_c), E3div));
            _mm_storel_epi64((__m128i*) rdata, _mm_packus_epi16(temp, _mm_setzero_si128()));
            rdata += Rimg->xsize;
        }

        for (size_t x = Simg->xsize - r; x < Simg->xsize; x += 1) {
            // X3.r += c[lastx & r_mask].r - c[(x-r) & r_mask].r;
            X3 = _mm_add_epi16(_mm_sub_epi16(X3, c[(x-r) & r_mask]), c[lastx & r_mask]);

            // *rdata = (uint8_t) ((X3.r * X3div + (c[(x-r) & r_mask].r + c[lastx & r_mask].r) * E3div + (1<<15)) >> 16);
            __m128i temp = _mm_add_epi16(_mm_mulhi_epu16(X3, X3div),
                _mm_mulhi_epu16(_mm_add_epi16(c[(x-r) & r_mask], c[lastx & r_mask]), E3div));
            _mm_storel_epi64((__m128i*) rdata, _mm_packus_epi16(temp, _mm_setzero_si128()));
            rdata += Rimg->xsize;
        }
    }
}


extern void
opTriBoxBlur_premul(
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
