#include <stdint.h>
#include <stddef.h>
#include <emmintrin.h>
#include <tmmintrin.h>
#include <xmmintrin.h>


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

        __m128i Rx2lo = _mm_maddubs_epi16(
            _mm_unpacklo_epi8(_mm_set1_epi8((char) 255), Sax4),
            _mm_unpacklo_epi8(Sx4, Dx4));
        __m128i Rx2hi = _mm_maddubs_epi16(
            _mm_unpackhi_epi8(_mm_set1_epi8((char) 255), Sax4),
            _mm_unpackhi_epi8(Sx4, Dx4));

        Rx2lo = _mm_add_epi16(Rx2lo, _mm_set1_epi16(0x80));
        Rx2lo = _mm_srli_epi16(_mm_add_epi16(_mm_srli_epi16(Rx2lo, 8), Rx2lo), 8);
        Rx2hi = _mm_add_epi16(Rx2hi, _mm_set1_epi16(0x80));
        Rx2hi = _mm_srli_epi16(_mm_add_epi16(_mm_srli_epi16(Rx2hi, 8), Rx2hi), 8);

        _mm_storeu_si128((__m128i*) &Rrgba[i], _mm_packus_epi16(Rx2lo, Rx2hi));
    }

    for (; i < len*4; i += 4) {
        __m128i Sx4 = (__m128i) _mm_load_ss((float*) &Srgba[i]);
        __m128i Dx4 = (__m128i) _mm_load_ss((float*) &Drgba[i]);
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

        _mm_store_ss((float*) &Rrgba[i], (__m128) _mm_packus_epi16(Rx2lo, Rx2lo));
    }
}
