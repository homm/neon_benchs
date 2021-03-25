#include <stdint.h>
#include <stddef.h>
#include <arm_neon.h>

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
        uint8x16_t Sx4 = vld1q_u8(&Srgba[i]);
        uint8x16_t Dx4 = vld1q_u8(&Drgba[i]);

        uint8x16_t Sax4 = vsubq_u8(
            vdupq_n_u8(255),
            vqtbl1q_u8(Sx4, (uint8x16_t){3,3,3,3, 7,7,7,7, 11,11,11,11, 15,15,15,15})
        );

        uint16x8_t Rx2lo = vmull_u8(vget_low_u8(Sx4), vdup_n_u8(255));
        uint16x8_t Rx2hi = vmull_high_u8(Sx4, vdupq_n_u8(255));

        Rx2lo = vmlal_u8(Rx2lo, vget_low_u8(Dx4), vget_low_u8(Sax4));
        Rx2hi = vmlal_high_u8(Rx2hi, Dx4, Sax4);

        uint8x16_t Rx4 = vqrshrn_high_n_u16(
            vqrshrn_n_u16(vrsraq_n_u16(Rx2lo, Rx2lo, 8), 8),
            vrsraq_n_u16(Rx2hi, Rx2hi, 8), 8);
        vst1q_u8(&Rrgba[i], Rx4);
    }

    for (; i < len*4; i += 4) {
        uint8x8_t Sx4 = (uint8x8_t)vld1_dup_u32((uint32_t *)&Srgba[i]);
        uint8x8_t Dx4 = (uint8x8_t)vld1_dup_u32((uint32_t *)&Drgba[i]);

        uint8x8_t Sax4 = vsub_u8(vdup_n_u8(255), vdup_lane_u8(Sx4, 3));

        uint16x8_t Rx2lo = vmull_u8(Sx4, vdup_n_u8(255));
        Rx2lo = vmlal_u8(Rx2lo, Dx4, Sax4);

        uint8x8_t Rx4 = vqrshrn_n_u16(vrsraq_n_u16(Rx2lo, Rx2lo, 8), 8);
        vst1_lane_u32((uint32_t *)&Rrgba[i], Rx4, 0);
    }
}
