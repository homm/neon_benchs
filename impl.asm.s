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
	b.eq	.LBB0_4
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
	ext	v5.16b, v2.16b, v2.16b, #8
	umull	v6.8h, v3.8b, v4.8b
	umull	v5.8h, v5.8b, v1.8b
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
	b.lo	.LBB0_5
	b	.LBB0_7
.LBB0_4:
	mov	x9, xzr
	cmp	x9, x8
	b.hs	.LBB0_7
.LBB0_5:
	movi	v0.2d, #0xffffffffffffffff
.LBB0_6:                                // =>This Inner Loop Header: Depth=1
	add	x10, x1, x9
	ld1r	{ v1.2s }, [x10]
	add	x10, x2, x9
	ld1r	{ v2.2s }, [x10]
	mvn	v3.8b, v1.8b
	dup	v3.8b, v3.b[3]
	umull	v2.8h, v2.8b, v3.8b
	umlal	v2.8h, v1.8b, v0.8b
	ursra	v2.8h, v2.8h, #8
	uqrshrn	v1.8b, v2.8h, #8
	str	s1, [x0, x9]
	add	x9, x9, #4              // =4
	cmp	x9, x8
	b.lo	.LBB0_6
.LBB0_7:
	ret
.Lfunc_end0:
	.size	opSourceOver_premul, .Lfunc_end0-opSourceOver_premul
                                        // -- End function

	.ident	"clang version 9.0.1-6+rpi1~bpo10+1 "
	.section	".note.GNU-stack","",@progbits
	.addrsig
