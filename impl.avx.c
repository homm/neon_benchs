#include <stdint.h>
#include <stddef.h>
#include <immintrin.h>


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

    for (; i < len*4 - 28; i += 32) {
        __m256i Sx4 = _mm256_loadu_si256((__m256i*) &Srgba[i]);
        __m256i Dx4 = _mm256_loadu_si256((__m256i*) &Drgba[i]);

        __m256i Sax4 = _mm256_sub_epi8(
            _mm256_set1_epi8((char) 255),
            _mm256_shuffle_epi8(Sx4, _mm256_set_epi8(
                15,15,15,15, 11,11,11,11, 7,7,7,7, 3,3,3,3,
                15,15,15,15, 11,11,11,11, 7,7,7,7, 3,3,3,3)));

        __m256i Rx2lo = _mm256_add_epi16(
            _mm256_mullo_epi16(_mm256_unpacklo_epi8(Sx4, _mm256_setzero_si256()),
                            _mm256_set1_epi16(255)),
            _mm256_mullo_epi16(_mm256_unpacklo_epi8(Dx4, _mm256_setzero_si256()),
                            _mm256_unpacklo_epi8(Sax4, _mm256_setzero_si256())));
        __m256i Rx2hi = _mm256_add_epi16(
            _mm256_mullo_epi16(_mm256_unpackhi_epi8(Sx4, _mm256_setzero_si256()),
                            _mm256_set1_epi16(255)),
            _mm256_mullo_epi16(_mm256_unpackhi_epi8(Dx4, _mm256_setzero_si256()),
                            _mm256_unpackhi_epi8(Sax4, _mm256_setzero_si256())));

        Rx2lo = _mm256_add_epi16(Rx2lo, _mm256_set1_epi16(0x80));
        Rx2lo = _mm256_srli_epi16(_mm256_add_epi16(_mm256_srli_epi16(Rx2lo, 8), Rx2lo), 8);
        Rx2hi = _mm256_add_epi16(Rx2hi, _mm256_set1_epi16(0x80));
        Rx2hi = _mm256_srli_epi16(_mm256_add_epi16(_mm256_srli_epi16(Rx2hi, 8), Rx2hi), 8);

        _mm256_storeu_si256((__m256i*) &Rrgba[i], _mm256_packus_epi16(Rx2lo, Rx2hi));
    }

    for (; i < len*4; i += 4) {
        uint8_t Sa = 255 - Srgba[i + 3];
        Rrgba[i + 0] = DIV255(Srgba[i + 0] * 255 + Drgba[i + 0] * Sa);
        Rrgba[i + 1] = DIV255(Srgba[i + 1] * 255 + Drgba[i + 1] * Sa);
        Rrgba[i + 2] = DIV255(Srgba[i + 2] * 255 + Drgba[i + 2] * Sa);
        Rrgba[i + 3] = DIV255(Srgba[i + 3] * 255 + Drgba[i + 3] * Sa);
    }
}
