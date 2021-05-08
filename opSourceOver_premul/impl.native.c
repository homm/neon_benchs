#include <stdint.h>
#include <stddef.h>

#define SHIFTFORDIV255(a)\
    ((((a) >> 8) + a) >> 8)

#define DIV255(a)\
    SHIFTFORDIV255(a + 0x80)

#ifndef REF
    #define REF
#endif

#define PASTER(x, y) x ## y
#define EVALUATOR(x, y)  PASTER(x, y)
#define DEFINE_REF(name) EVALUATOR(name, REF)


extern void
DEFINE_REF(opSourceOver_premul)(
    uint8_t* restrict Rrgba, const uint8_t* restrict Srgba,
    const uint8_t* restrict Drgba, size_t len)
{
    size_t i = 0;
    for (; i < len*4; i += 4) {
        uint8_t Sa = 255 - Srgba[i + 3];
        Rrgba[i + 0] = DIV255(Srgba[i + 0] * 255 + Drgba[i + 0] * Sa);
        Rrgba[i + 1] = DIV255(Srgba[i + 1] * 255 + Drgba[i + 1] * Sa);
        Rrgba[i + 2] = DIV255(Srgba[i + 2] * 255 + Drgba[i + 2] * Sa);
        Rrgba[i + 3] = DIV255(Srgba[i + 3] * 255 + Drgba[i + 3] * Sa);
    }
}

/* extern void
opSourceOver(uint8_t *Rrgba, uint8_t *Srgba, uint8_t *Drgba, size_t len)
{
    size_t i = 0;
    for (; i < len*4 - 12; i += 16) {
        uint8x16_t Sx4 = vld1q_u8(&Srgba[i]);
        uint8x16_t Dx4 = vld1q_u8(&Drgba[i]);
        uint32x4_t Sax4 = vshrq_n_u32((uint32x4_t) Sx4, 24);
        uint32x4_t Dax4 = vshrq_n_u32((uint32x4_t) Dx4, 24);

        uint32x4_t blendx4 = vmulq_u32(Dax4, vsubq_u32(vdupq_n_u32(255), Sax4));
        uint32x4_t Ra255x4 = vaddq_u32(vmulq_u32(Sax4, vdupq_n_u32(255)), blendx4);

        uint32x4_t c1x4 = vmulq_u32(Sax4, vdupq_n_u32(255 * 255 << PRECISION_BITS));
        c1x4 = vcvtq_u32_f32(vdivq_f32(vcvtq_f32_u32(c1x4), vcvtq_f32_u32(Ra255x4)));

        vst1q_u8(&Rrgba[i], vaddq_u8(Sx4, (uint8x16_t) c1x4));
    }

    for (; i < len*4; i += 4) {
        uint8_t Sa = Srgba[i + 3];
        uint8_t Da = Drgba[i + 3];
        if (Sa == 0) {
            // Copy 4 bytes at once.
            memcpy(&Rrgba[i], &Drgba[i], 4);
        } else {
            // Integer implementation with increased precision.
            // Each variable has extra meaningful bits.
            // Divisions are rounded.

            uint16_t blend = Da * (255 - Sa);  // 65025
            uint16_t Ra255 = Sa * 255 + blend;  // 65025
            // There we use 7 bits for precision.
            // We could use more, but we go beyond 32 bits.
            uint16_t c1 = Sa * (255 * 255 << PRECISION_BITS) / Ra255;  // 32640
            uint16_t c2 = (255 << PRECISION_BITS) - c1;  // 32640

            uint32_t Rr = Srgba[i + 0] * c1 + Drgba[i + 0] * c2;
            uint32_t Rg = Srgba[i + 1] * c1 + Drgba[i + 1] * c2;
            uint32_t Rb = Srgba[i + 2] * c1 + Drgba[i + 2] * c2;
            Rrgba[i + 0] = SHIFTFORDIV255(Rr + (0x80<<PRECISION_BITS)) >> PRECISION_BITS;
            Rrgba[i + 1] = SHIFTFORDIV255(Rg + (0x80<<PRECISION_BITS)) >> PRECISION_BITS;
            Rrgba[i + 2] = SHIFTFORDIV255(Rb + (0x80<<PRECISION_BITS)) >> PRECISION_BITS;
            Rrgba[i + 3] = SHIFTFORDIV255(Ra255 + 0x80);
        }
    }
} */
