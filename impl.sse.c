#include <stdint.h>
#include <stddef.h>
#include <emmintrin.h>
#include <tmmintrin.h>

#define SHIFTFORDIV255(a)\
    ((((a) >> 8) + a) >> 8)

#define DIV255(a)\
    SHIFTFORDIV255(a + 0x80)


extern void
opSourceOver_premul(uint8_t* restrict Rrgba,
                    const uint8_t* restrict Srgba,
                    const uint8_t* restrict Drgba, size_t len)
{
    size_t i = 0;

    for (; i < len*4 - 12; i += 16) {
        __m128i Sx4 = _mm_loadu_si128((__m128i*) &Srgba[i]);
        __m128i Dx4 = _mm_loadu_si128((__m128i*) &Drgba[i]);
        __m128i Sax4 = _mm_sub_epi8(
            _mm_set1_epi8((char) 255),
            _mm_shuffle_epi8(Sx4, _mm_set_epi8(
                15,15,15,15, 11,11,11,11, 7,7,7,7, 3,3,3,3)));

        __m128i Rx2lo = _mm_add_epi16(
            _mm_mullo_epi16(_mm_unpacklo_epi8(Sx4, _mm_setzero_si128()),
                            _mm_set1_epi16(255)),
            _mm_mullo_epi16(_mm_unpacklo_epi8(Dx4, _mm_setzero_si128()),
                            _mm_unpacklo_epi8(Sax4, _mm_setzero_si128())));
        Rx2lo = _mm_add_epi16(Rx2lo, _mm_set1_epi16(0x80));
        Rx2lo = _mm_srli_epi16(_mm_add_epi16(_mm_srli_epi16(Rx2lo, 8), Rx2lo), 8);

        __m128i Rx2hi = _mm_add_epi16(
            _mm_mullo_epi16(_mm_unpackhi_epi8(Sx4, _mm_setzero_si128()),
                            _mm_set1_epi16(255)),
            _mm_mullo_epi16(_mm_unpackhi_epi8(Dx4, _mm_setzero_si128()),
                            _mm_unpackhi_epi8(Sax4, _mm_setzero_si128())));
        Rx2hi = _mm_add_epi16(Rx2hi, _mm_set1_epi16(0x80));
        Rx2hi = _mm_srli_epi16(_mm_add_epi16(_mm_srli_epi16(Rx2hi, 8), Rx2hi), 8);

        _mm_storeu_si128((__m128i*) &Rrgba[i], _mm_packus_epi16(Rx2lo, Rx2hi));
    }

    for (; i < len*4; i += 4) {
        uint8_t Sa = 255 - Srgba[i + 3];
        Rrgba[i + 0] = DIV255(Srgba[i + 0] * 255 + Drgba[i + 0] * Sa);
        Rrgba[i + 1] = DIV255(Srgba[i + 1] * 255 + Drgba[i + 1] * Sa);
        Rrgba[i + 2] = DIV255(Srgba[i + 2] * 255 + Drgba[i + 2] * Sa);
        Rrgba[i + 3] = DIV255(Srgba[i + 3] * 255 + Drgba[i + 3] * Sa);
    }
}
