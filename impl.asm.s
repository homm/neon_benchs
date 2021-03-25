	.text
	.file	"impl.preload.c"
	.section	.rodata.cst16,"aM",@progbits,16
	.p2align	4               // -- Begin function opSourceOver_premul
.LCPI0_0:
	.byte	3                       // 0x3
	.byte	3                       // 0x3
	.byte	3                       // 0x3
	.byte	3                       // 0x3
	.byte	7                       // 0x7
	.byte	7                       // 0x7
	.byte	7                       // 0x7
	.byte	7                       // 0x7
	.byte	11                      // 0xb
	.byte	11                      // 0xb
	.byte	11                      // 0xb
	.byte	11                      // 0xb
	.byte	15                      // 0xf
	.byte	15                      // 0xf
	.byte	15                      // 0xf
	.byte	15                      // 0xf
	.text
	.globl	opSourceOver_premul
	.p2align	2
	.type	opSourceOver_premul,@function
opSourceOver_premul:                    // @opSourceOver_premul
// %bb.0:
	lsl	x8, x3, #2
	subs	x10, x8, #28            // =28
	b.eq	.LBB0_6
// %bb.1:
	mov	x11, x2
	mov	x13, x1
	adrp	x9, .LCPI0_0
	ldr	q3, [x11], #16
	ldr	q2, [x13], #16
	ldr	q0, [x9, :lo12:.LCPI0_0]
	mov	x12, xzr
	movi	v1.2d, #0xffffffffffffffff

.LBB0_2:                                // =>This Inner Loop Header: Depth=1
	tbl	v4.16b, { v2.16b }, v0.16b
	mvn	v4.16b, v4.16b
	umull	v6.8h, v3.8b, v4.8b
	umull2	v5.8h, v2.16b, v1.16b
	umlal	v6.8h, v2.8b, v1.8b
	umlal2	v5.8h, v3.16b, v4.16b
	ldr	q3, [x11, x12]
	ldr	q2, [x13, x12]
	ursra	v6.8h, v6.8h, #8
	ursra	v5.8h, v5.8h, #8
	uqrshrn	v4.8b, v6.8h, #8
	add	x9, x12, #16            // =16
	uqrshrn2	v4.16b, v5.8h, #8
	cmp	x9, x10
	str	q4, [x0, x12]
	mov	x12, x9
	b.lo	.LBB0_2

// %bb.3:
	cmp	x9, x8
	b.hs	.LBB0_5
.LBB0_4:                                // =>This Inner Loop Header: Depth=1
	orr	x10, x9, #0x3
	ldrb	w11, [x1, x9]
	orr	x13, x9, #0x1
	ldrb	w15, [x1, x10]
	ldrb	w12, [x2, x9]
	orr	x14, x9, #0x2
	ldrb	w16, [x1, x13]
	ldrb	w18, [x1, x14]
	lsl	w5, w11, #8
	sub	w11, w5, w11
	eor	w5, w15, #0xff
	ldrb	w17, [x2, x13]
	ldrb	w3, [x2, x14]
	ldrb	w4, [x2, x10]
	madd	w11, w12, w5, w11
	lsl	w12, w16, #8
	sub	w12, w12, w16
	lsl	w16, w18, #8
	sub	w16, w16, w18
	lsl	w18, w15, #8
	add	w11, w11, #127          // =127
	sub	w15, w18, w15
	add	w11, w11, w11, lsr #8
	madd	w12, w17, w5, w12
	madd	w16, w3, w5, w16
	madd	w15, w4, w5, w15
	lsr	w11, w11, #8
	strb	w11, [x0, x9]
	add	w11, w12, #127          // =127
	add	w12, w16, #127          // =127
	add	w15, w15, #127          // =127
	add	w11, w11, w11, lsr #8
	add	w12, w12, w12, lsr #8
	add	w15, w15, w15, lsr #8
	add	x9, x9, #4              // =4
	lsr	w11, w11, #8
	lsr	w12, w12, #8
	lsr	w15, w15, #8
	cmp	x9, x8
	strb	w11, [x0, x13]
	strb	w12, [x0, x14]
	strb	w15, [x0, x10]
	b.lo	.LBB0_4
.LBB0_5:
	ret
.LBB0_6:
	mov	x9, xzr
	cmp	x9, x8
	b.lo	.LBB0_4
	b	.LBB0_5
.Lfunc_end0:
	.size	opSourceOver_premul, .Lfunc_end0-opSourceOver_premul
                                        // -- End function

	.ident	"clang version 9.0.1-6+rpi1~bpo10+1 "
	.section	".note.GNU-stack","",@progbits
	.addrsig
