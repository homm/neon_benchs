    .text
    .file   "impl.preload.c"
    .section    .rodata.cst16,"aM",@progbits,16
    .p2align    4               // -- Begin function opSourceOver_premul
.LCPI0_0:
    .byte   3                       // 0x3
    .byte   3                       // 0x3
    .byte   3                       // 0x3
    .byte   3                       // 0x3
    .byte   7                       // 0x7
    .byte   7                       // 0x7
    .byte   7                       // 0x7
    .byte   7                       // 0x7
    .byte   11                      // 0xb
    .byte   11                      // 0xb
    .byte   11                      // 0xb
    .byte   11                      // 0xb
    .byte   15                      // 0xf
    .byte   15                      // 0xf
    .byte   15                      // 0xf
    .byte   15                      // 0xf
    .text
    .globl  opSourceOver_premul
    .p2align    2
    .type   opSourceOver_premul,@function
opSourceOver_premul:                    // @opSourceOver_premul
// %bb.0:
    lsl x8, x3, #2
    subs    x12, x8, #28            // =28
    b.eq    .LBB0_4
// %bb.1:
    adrp    x9, .LCPI0_0
    ldr q16, [x9, :lo12:.LCPI0_0]

    ldr     q1, [x1]                    // Sx4 = vld1q_u8(&Srgba[0])
    ldr     q2, [x2]                    // Dx4 = vld1q_u8(&Drgba[0])
    movi    v17.2d, #0xffffffffffffffff
    mov     x9, xzr                     // i = 0
.LBB0_2:
    tbl     v3.16b, { v1.16b }, v16.16b // vqtbl1q_u8(Sx4, v16)
    mvn     v3.16b, v3.16b              // Sax4 = vsubq_u8(vdupq_n_u8(255), v3)
    add     x10, x9, #16                // x10 = i + 16
    umull   v5.8h, v1.8b,  v17.8b       // Rx2lo = vmull_u8(Sx4, 0xff);
    umull2  v6.8h, v1.16b, v17.16b      // Rx2hi = vmull_high_u8(Sx4, 0xff)
    ldr     q1, [x1, x10]               // Sx4 = vld1q_u8(&Srgba[i + 16])
    umlal   v5.8h, v2.8b,  v3.8b        // vmlal_u8(Rx2lo, Dx4, Sax4)
    umlal2  v6.8h, v2.16b, v3.16b       // vmlal_high_u8(Rx2hi, Dx4, Sax4)
    ldr     q2, [x2, x10]               // Dx4 = vld1q_u8(&Drgba[i + 16])
    ursra   v5.8h, v5.8h, #8            // vrsraq_n_u16(Rx2lo, Rx2lo, 8)
    ursra   v6.8h, v6.8h, #8            // vrsraq_n_u16(Rx2hi, Rx2hi, 8)
    uqrshrn v0.8b, v5.8h, #8            // Rx4 = vqrshrn_n_u16(Rx2lo, 8)
    uqrshrn2 v0.16b, v6.8h, #8          // vqrshrn_high_n_u16(Rx4, Rx2hi, 8)
    cmp     x10, x12
    str     q0, [x0, x9]                // vst1q_u8(&Rrgba[i], Rx4)
    mov     x9, x10
    b.lo    .LBB0_2


// %bb.3:
    cmp x9, x8
    b.lo    .LBB0_5
    b   .LBB0_7
.LBB0_4:
    mov x9, xzr
    cmp x9, x8
    b.hs    .LBB0_7
.LBB0_5:
    movi    v0.2d, #0xffffffffffffffff
.LBB0_6:                                // =>This Inner Loop Header: Depth=1
    add x10, x1, x9
    ld1r    { v1.2s }, [x10]
    add x10, x2, x9
    ld1r    { v2.2s }, [x10]
    mvn v3.8b, v1.8b
    dup v3.8b, v3.b[3]
    umull   v2.8h, v2.8b, v3.8b
    umlal   v2.8h, v1.8b, v0.8b
    ursra   v2.8h, v2.8h, #8
    uqrshrn v1.8b, v2.8h, #8
    str s1, [x0, x9]
    add x9, x9, #4              // =4
    cmp x9, x8
    b.lo    .LBB0_6
.LBB0_7:
    ret
.Lfunc_end0:
    .size   opSourceOver_premul, .Lfunc_end0-opSourceOver_premul
                                        // -- End function

    .ident  "clang version 9.0.1-6+rpi1~bpo10+1 "
    .section    ".note.GNU-stack","",@progbits
    .addrsig
