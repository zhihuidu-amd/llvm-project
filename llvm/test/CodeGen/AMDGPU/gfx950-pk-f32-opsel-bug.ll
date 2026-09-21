; RUN: llc -mtriple=amdgpu9.42-- -global-isel=0 < %s | FileCheck -check-prefix=GFX942 %s
; RUN: llc -mtriple=amdgpu9.50-- -global-isel=0 < %s | FileCheck -check-prefix=GFX950 --implicit-check-not=op_sel:[0,1 %s
; RUN: llc -mtriple=amdgpu9.50-- -global-isel=0 -O0 < %s | FileCheck -check-prefix=O0-950 --implicit-check-not=op_sel:[0,1 %s
; RUN: llc -mtriple=amdgpu9.50-- -global-isel=1 < %s | FileCheck -check-prefix=GI-950 --implicit-check-not=op_sel:[0,1 %s

; GFX950 silently miscomputes the low half of a VOP3P packed-f32 instruction
; that gathers src0's low dword with src1's high dword -- op_sel:[0,1] -- when
; MFMA work is in flight on the same CU: the src1 operand can read as zero.
; Nothing faults, the sources are undisturbed and the high half is always
; correct, so the only symptom is a wrong result.
;
; GCNHazardRecognizer::fixPkF32OpSelBug avoids the encoding by commuting the
; operands. Each source's op_sel, op_sel_hi, neg_lo and neg_hi bits live in that
; source's own modifier operand, so the existing commute carries every modifier
; along with the operand it belongs to: op_sel:[0,1] becomes op_sel:[1,0], the
; arithmetic is bit-for-bit identical, and no instruction or register is added.
;
; gfx942 is the negative control. The predicate is off there, so the same IR
; must still produce op_sel:[0,1]; if a future change stopped generating the
; pattern at all, the GFX942 lines would fail rather than letting this test
; quietly degrade into checking nothing.
;
; The -O0 RUN line is not redundant. SIPreEmitPeephole -- which already hosts a
; gfx940+ packed-f32 transform and is the obvious place to put this -- is added
; only under "getOptLevel() > CodeGenOptLevel::None", so a workaround living
; there would not run at -O0 and the defective encoding would still ship. This
; one runs from PostRAHazardRecognizer, which is added unconditionally.

; GFX942: v_pk_add_f32 v[{{[0-9:]+}}], v[{{[0-9:]+}}], v[{{[0-9:]+}}] op_sel:[0,1] op_sel_hi:[1,0]
; GFX950: v_pk_add_f32 v[{{[0-9:]+}}], v[{{[0-9:]+}}], v[{{[0-9:]+}}] op_sel:[1,0] op_sel_hi:[0,1]
define amdgpu_kernel void @pk_add_opsel_01(ptr addrspace(1) %out, ptr addrspace(3) %lds) {
bb:
  %a = load volatile <2 x float>, ptr addrspace(3) %lds, align 8
  %gep = getelementptr inbounds <2 x float>, ptr addrspace(3) %lds, i32 1
  %b = load volatile <2 x float>, ptr addrspace(3) %gep, align 8
  %b.swap = shufflevector <2 x float> %b, <2 x float> poison, <2 x i32> <i32 1, i32 0>
  %r = fadd <2 x float> %a, %b.swap
  store <2 x float> %r, ptr addrspace(1) %out, align 8
  ret void
}

; GFX942: v_pk_mul_f32 v[{{[0-9:]+}}], v[{{[0-9:]+}}], v[{{[0-9:]+}}] op_sel:[0,1] op_sel_hi:[1,0]
; GFX950: v_pk_mul_f32 v[{{[0-9:]+}}], v[{{[0-9:]+}}], v[{{[0-9:]+}}] op_sel:[1,0] op_sel_hi:[0,1]
define amdgpu_kernel void @pk_mul_opsel_01(ptr addrspace(1) %out, ptr addrspace(3) %lds) {
bb:
  %a = load volatile <2 x float>, ptr addrspace(3) %lds, align 8
  %gep = getelementptr inbounds <2 x float>, ptr addrspace(3) %lds, i32 1
  %b = load volatile <2 x float>, ptr addrspace(3) %gep, align 8
  %b.swap = shufflevector <2 x float> %b, <2 x float> poison, <2 x i32> <i32 1, i32 0>
  %r = fmul <2 x float> %a, %b.swap
  store <2 x float> %r, ptr addrspace(1) %out, align 8
  ret void
}

; The shape that actually ships: a complex multiply-accumulate. One product is
; negated and the real/imaginary halves are crossed, so the instruction carries
; neg_lo and neg_hi as well as op_sel and op_sel_hi. All four pairs of bits have
; to move with their operand, which is what this function pins down. Both the
; gfx942 and the gfx950 form below were run on hardware: the gfx942 form fails
; 20000/20000 launches and the gfx950 form 0/20000.
;
; GFX942: v_pk_fma_f32 v[{{[0-9:]+}}], v[{{[0-9:]+}}], v[{{[0-9:]+}}], v[{{[0-9:]+}}] op_sel:[0,1,0] op_sel_hi:[1,0,1] neg_lo:[1,0,0] neg_hi:[1,0,0]
; GFX950: v_pk_fma_f32 v[{{[0-9:]+}}], v[{{[0-9:]+}}], v[{{[0-9:]+}}], v[{{[0-9:]+}}] op_sel:[1,0,0] op_sel_hi:[0,1,1] neg_lo:[0,1,0] neg_hi:[0,1,0]
define amdgpu_kernel void @pk_fma_opsel_01_neg(ptr addrspace(1) %out, ptr addrspace(3) %lds) {
bb:
  %a = load volatile <2 x float>, ptr addrspace(3) %lds, align 8
  %gep1 = getelementptr inbounds <2 x float>, ptr addrspace(3) %lds, i32 1
  %b = load volatile <2 x float>, ptr addrspace(3) %gep1, align 8
  %gep2 = getelementptr inbounds <2 x float>, ptr addrspace(3) %lds, i32 2
  %c = load volatile <2 x float>, ptr addrspace(3) %gep2, align 8
  %b.swap = shufflevector <2 x float> %b, <2 x float> poison, <2 x i32> <i32 1, i32 0>
  %a.neg = fneg <2 x float> %a
  %r = call <2 x float> @llvm.fma.v2f32(<2 x float> %a.neg, <2 x float> %b.swap, <2 x float> %c)
  store <2 x float> %r, ptr addrspace(1) %out, align 8
  ret void
}

; An SGPR source. Commuting moves the scalar operand into src0, which is still
; legal for VOP3P; this is the operand class most likely to make
; commuteInstruction refuse, and fixPkF32OpSelBug reports a fatal error rather
; than emit the defective encoding if it ever does.
;
; An SGPR src1 did not reproduce the defect on hardware -- 0/20000 launches,
; against 20000/20000 for the VGPR form in the same job on the same GPU -- so
; rewriting this one is conservative rather than required. It stays covered
; because the rewrite is free and the scalar case was measured on one shape only.
;
; GFX942: v_pk_add_f32 v[{{[0-9:]+}}], v[{{[0-9:]+}}], s[{{[0-9:]+}}] op_sel:[0,1] op_sel_hi:[1,0]
; GFX950: v_pk_add_f32 v[{{[0-9:]+}}], s[{{[0-9:]+}}], v[{{[0-9:]+}}] op_sel:[1,0] op_sel_hi:[0,1]
define amdgpu_kernel void @pk_add_opsel_01_sgpr(ptr addrspace(1) %out, <2 x float> inreg %s, ptr addrspace(3) %lds) {
bb:
  %a = load volatile <2 x float>, ptr addrspace(3) %lds, align 8
  %s.swap = shufflevector <2 x float> %s, <2 x float> poison, <2 x i32> <i32 1, i32 0>
  %r = fadd <2 x float> %a, %s.swap
  store <2 x float> %r, ptr addrspace(1) %out, align 8
  ret void
}

; Negative control for the shape condition. op_sel:[1,1] takes both high dwords
; and measured clean (0/2000 launches), so op_sel[1] == 1 is not by itself the
; trigger and this must be left alone on gfx950. Written as a predicate of the
; form "op_sel[1] == 1" the workaround would rewrite this too; the O0-950 line
; below is where that shows up, because at -O1 and above the shuffle is folded
; away before it reaches the hazard recognizer.
;
; O0-950: v_pk_add_f32 v[{{[0-9:]+}}], v[{{[0-9:]+}}], v[{{[0-9:]+}}] op_sel:[1,1]{{$}}
define amdgpu_kernel void @pk_add_opsel_11_untouched(ptr addrspace(1) %out, ptr addrspace(3) %lds) {
bb:
  %a = load volatile <2 x float>, ptr addrspace(3) %lds, align 8
  %gep = getelementptr inbounds <2 x float>, ptr addrspace(3) %lds, i32 1
  %b = load volatile <2 x float>, ptr addrspace(3) %gep, align 8
  %a.hi = shufflevector <2 x float> %a, <2 x float> poison, <2 x i32> <i32 1, i32 1>
  %b.hi = shufflevector <2 x float> %b, <2 x float> poison, <2 x i32> <i32 1, i32 1>
  %r = fadd <2 x float> %a.hi, %b.hi
  store <2 x float> %r, ptr addrspace(1) %out, align 8
  ret void
}

; GlobalISel selects different shapes than SelectionDAG -- most of these become
; a plain packed op against the SGPR pair with no op_sel at all -- so it gets a
; prefix of its own instead of the per-function checks above. What matters is
; the same thing in both: the --implicit-check-not on every gfx950 RUN line
; means no output from this file may carry the defective encoding, whatever the
; instruction selector and register allocator decide to do.
;
; GI-950: v_pk_add_f32 v[{{[0-9:]+}}], s[{{[0-9:]+}}], v[{{[0-9:]+}}] op_sel:[1,0] op_sel_hi:[0,1]

declare <2 x float> @llvm.fma.v2f32(<2 x float>, <2 x float>, <2 x float>)
