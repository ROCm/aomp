; ModuleID = 'kernel-hip-amdgcn-amd-amdhsa-gfx906.bc'
source_filename = "kernel-hip-amdgcn-amd-amdhsa-gfx906.cui"
target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-ni:7"
target triple = "amdgcn-amd-amdhsa"

%struct.__hip_builtin_blockIdx_t = type { i8 }

$_ZN24__hip_builtin_blockIdx_t7__get_xEv = comdat any

@__hip_device_heap = weak protected addrspace(1) externally_initialized global [4194304 x i8] zeroinitializer, align 16
@__hip_device_page_flag = weak protected addrspace(1) externally_initialized global [65536 x i32] zeroinitializer, align 16
@blockIdx = extern_weak hidden addrspace(1) global %struct.__hip_builtin_blockIdx_t, align 1

; Function Attrs: convergent nounwind
define protected amdgpu_kernel void @_Z10writeIndexPii(i32 addrspace(1)* %b, i32 %n) #0 {
entry:
  %b.addr = alloca i32 addrspace(1)*, align 8, addrspace(5)
  %b.addr.ascast = addrspacecast i32 addrspace(1)* addrspace(5)* %b.addr to i32 addrspace(1)**
  %n.addr = alloca i32, align 4, addrspace(5)
  %n.addr.ascast = addrspacecast i32 addrspace(5)* %n.addr to i32*
  %intpart = alloca float, align 4, addrspace(5)
  %intpart.ascast = addrspacecast float addrspace(5)* %intpart to float*
  %res = alloca float, align 4, addrspace(5)
  %res.ascast = addrspacecast float addrspace(5)* %res to float*
  %i = alloca i32, align 4, addrspace(5)
  %i.ascast = addrspacecast i32 addrspace(5)* %i to i32*
  store i32 addrspace(1)* %b, i32 addrspace(1)** %b.addr.ascast, align 8, !tbaa !4
  store i32 %n, i32* %n.addr.ascast, align 4, !tbaa !8
  %0 = bitcast float addrspace(5)* %intpart to i8 addrspace(5)*
  call void @llvm.lifetime.start.p5i8(i64 4, i8 addrspace(5)* %0) #7
  %1 = bitcast float addrspace(5)* %res to i8 addrspace(5)*
  call void @llvm.lifetime.start.p5i8(i64 4, i8 addrspace(5)* %1) #7
  %call = call float @_ZL5modfffPf(float 0x3FF19999A0000000, float* %intpart.ascast) #8
  store float %call, float* %res.ascast, align 4, !tbaa !10
  %2 = bitcast i32 addrspace(5)* %i to i8 addrspace(5)*
  call void @llvm.lifetime.start.p5i8(i64 4, i8 addrspace(5)* %2) #7
  %call1 = call i32 @_ZN24__hip_builtin_blockIdx_t7__get_xEv() #8
  store i32 %call1, i32* %i.ascast, align 4, !tbaa !8
  %3 = load i32, i32* %i.ascast, align 4, !tbaa !8
  %4 = load i32, i32* %n.addr.ascast, align 4, !tbaa !8
  %cmp = icmp slt i32 %3, %4
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %5 = load float, float* %intpart.ascast, align 4, !tbaa !10
  %conv = fptosi float %5 to i32
  %6 = load i32 addrspace(1)*, i32 addrspace(1)** %b.addr.ascast, align 8, !tbaa !4
  %7 = load i32, i32* %i.ascast, align 4, !tbaa !8
  %idxprom = sext i32 %7 to i64
  %arrayidx = getelementptr inbounds i32, i32 addrspace(1)* %6, i64 %idxprom
  store i32 %conv, i32 addrspace(1)* %arrayidx, align 4, !tbaa !8
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  %8 = bitcast i32 addrspace(5)* %i to i8 addrspace(5)*
  call void @llvm.lifetime.end.p5i8(i64 4, i8 addrspace(5)* %8) #7
  %9 = bitcast float addrspace(5)* %res to i8 addrspace(5)*
  call void @llvm.lifetime.end.p5i8(i64 4, i8 addrspace(5)* %9) #7
  %10 = bitcast float addrspace(5)* %intpart to i8 addrspace(5)*
  call void @llvm.lifetime.end.p5i8(i64 4, i8 addrspace(5)* %10) #7
  ret void
}

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start.p5i8(i64 immarg, i8 addrspace(5)* nocapture) #1

; Function Attrs: convergent inlinehint nounwind
define internal float @_ZL5modfffPf(float %x, float* %iptr) #2 {
entry:
  %retval = alloca float, align 4, addrspace(5)
  %retval.ascast = addrspacecast float addrspace(5)* %retval to float*
  %x.addr = alloca float, align 4, addrspace(5)
  %x.addr.ascast = addrspacecast float addrspace(5)* %x.addr to float*
  %iptr.addr = alloca float*, align 8, addrspace(5)
  %iptr.addr.ascast = addrspacecast float* addrspace(5)* %iptr.addr to float**
  %tmp = alloca float, align 4, addrspace(5)
  %tmp.ascast = addrspacecast float addrspace(5)* %tmp to float*
  %r = alloca float, align 4, addrspace(5)
  %r.ascast = addrspacecast float addrspace(5)* %r to float*
  %cleanup.dest.slot = alloca i32, align 4, addrspace(5)
  store float %x, float* %x.addr.ascast, align 4, !tbaa !10
  store float* %iptr, float** %iptr.addr.ascast, align 8, !tbaa !4
  %0 = bitcast float addrspace(5)* %tmp to i8 addrspace(5)*
  call void @llvm.lifetime.start.p5i8(i64 4, i8 addrspace(5)* %0) #7
  %1 = bitcast float addrspace(5)* %r to i8 addrspace(5)*
  call void @llvm.lifetime.start.p5i8(i64 4, i8 addrspace(5)* %1) #7
  %2 = load float, float* %x.addr.ascast, align 4, !tbaa !10
  %tmp.ascast.ascast = addrspacecast float* %tmp.ascast to float addrspace(5)*
  %call = call float @__ocml_modf_f32(float %2, float addrspace(5)* %tmp.ascast.ascast) #8
  store float %call, float* %r.ascast, align 4, !tbaa !10
  %3 = load float, float* %tmp.ascast, align 4, !tbaa !10
  %4 = load float*, float** %iptr.addr.ascast, align 8, !tbaa !4
  store float %3, float* %4, align 4, !tbaa !10
  %5 = load float, float* %r.ascast, align 4, !tbaa !10
  %6 = bitcast float addrspace(5)* %r to i8 addrspace(5)*
  call void @llvm.lifetime.end.p5i8(i64 4, i8 addrspace(5)* %6) #7
  %7 = bitcast float addrspace(5)* %tmp to i8 addrspace(5)*
  call void @llvm.lifetime.end.p5i8(i64 4, i8 addrspace(5)* %7) #7
  ret float %5
}

; Function Attrs: alwaysinline convergent nounwind
define linkonce_odr hidden i32 @_ZN24__hip_builtin_blockIdx_t7__get_xEv() #3 comdat align 2 {
entry:
  %retval = alloca i32, align 4, addrspace(5)
  %retval.ascast = addrspacecast i32 addrspace(5)* %retval to i32*
  %call = call i32 @_ZL21__hip_get_block_idx_xv() #8
  ret i32 %call
}

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end.p5i8(i64 immarg, i8 addrspace(5)* nocapture) #1

; Function Attrs: alwaysinline convergent nounwind
define internal i32 @_ZL21__hip_get_block_idx_xv() #3 {
entry:
  %retval = alloca i32, align 4, addrspace(5)
  %retval.ascast = addrspacecast i32 addrspace(5)* %retval to i32*
  %call = call i64 @__ockl_get_group_id(i32 0) #8
  %conv = trunc i64 %call to i32
  ret i32 %conv
}

; Function Attrs: convergent nofree nounwind writeonly
define internal float @__ocml_modf_f32(float, float addrspace(5)* nocapture) #4 {
  %3 = tail call float @llvm.trunc.f32(float %0)
  %4 = fsub float %0, %3
  %5 = tail call i1 @llvm.amdgcn.class.f32(float %0, i32 516)
  %6 = select i1 %5, float 0.000000e+00, float %4
  store float %3, float addrspace(5)* %1, align 4, !tbaa !12
  %7 = tail call float @llvm.copysign.f32(float %6, float %0)
  ret float %7
}

; Function Attrs: nounwind readnone speculatable
declare float @llvm.trunc.f32(float) #5

; Function Attrs: nounwind readnone speculatable
declare i1 @llvm.amdgcn.class.f32(float, i32) #5

; Function Attrs: nounwind readnone speculatable
declare float @llvm.copysign.f32(float, float) #5

; Function Attrs: convergent nounwind readnone
define internal i64 @__ockl_get_group_id(i32) #6 {
  switch i32 %0, label %8 [
    i32 0, label %2
    i32 1, label %4
    i32 2, label %6
  ]

2:                                                ; preds = %1
  %3 = tail call i32 @llvm.amdgcn.workgroup.id.x()
  br label %8

4:                                                ; preds = %1
  %5 = tail call i32 @llvm.amdgcn.workgroup.id.y()
  br label %8

6:                                                ; preds = %1
  %7 = tail call i32 @llvm.amdgcn.workgroup.id.z()
  br label %8

8:                                                ; preds = %6, %4, %2, %1
  %9 = phi i32 [ %7, %6 ], [ %5, %4 ], [ %3, %2 ], [ 0, %1 ]
  %10 = zext i32 %9 to i64
  ret i64 %10
}

; Function Attrs: nounwind readnone speculatable
declare i32 @llvm.amdgcn.workgroup.id.x() #5

; Function Attrs: nounwind readnone speculatable
declare i32 @llvm.amdgcn.workgroup.id.y() #5

; Function Attrs: nounwind readnone speculatable
declare i32 @llvm.amdgcn.workgroup.id.z() #5

attributes #0 = { convergent nounwind "amdgpu-implicitarg-num-bytes"="56" "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="gfx906" "target-features"="+16-bit-insts,+ci-insts,+dl-insts,+dot1-insts,+dot2-insts,+dpp,+fp32-denormals,+fp64-fp16-denormals,+gfx8-insts,+gfx9-insts,+s-memrealtime" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { argmemonly nounwind }
attributes #2 = { convergent inlinehint nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="gfx906" "target-features"="+16-bit-insts,+ci-insts,+dl-insts,+dot1-insts,+dot2-insts,+dpp,+fp32-denormals,+fp64-fp16-denormals,+gfx8-insts,+gfx9-insts,+s-memrealtime" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { alwaysinline convergent nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="gfx906" "target-features"="+16-bit-insts,+ci-insts,+dl-insts,+dot1-insts,+dot2-insts,+dpp,+fp32-denormals,+fp64-fp16-denormals,+gfx8-insts,+gfx9-insts,+s-memrealtime" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { convergent nofree nounwind writeonly "correctly-rounded-divide-sqrt-fp-math"="false" "denorms-are-zero"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-features"="+fp64-fp16-denormals,-fp32-denormals" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #5 = { nounwind readnone speculatable }
attributes #6 = { convergent nounwind readnone "correctly-rounded-divide-sqrt-fp-math"="false" "denorms-are-zero"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-features"="+fp64-fp16-denormals,-fp32-denormals" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #7 = { nounwind }
attributes #8 = { convergent nounwind }

!llvm.module.flags = !{!0, !1}
!llvm.ident = !{!2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2}
!opencl.ocl.version = !{!3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"PIC Level", i32 1}
!2 = !{!"clang version 9.0.0 (https://github.com/ROCm-Developer-Tools/llvm-project 8fd8c0669dc9af4b909a98650ec542ac89fa72d1)"}
!3 = !{i32 2, i32 0}
!4 = !{!5, !5, i64 0}
!5 = !{!"any pointer", !6, i64 0}
!6 = !{!"omnipotent char", !7, i64 0}
!7 = !{!"Simple C++ TBAA"}
!8 = !{!9, !9, i64 0}
!9 = !{!"int", !6, i64 0}
!10 = !{!11, !11, i64 0}
!11 = !{!"float", !6, i64 0}
!12 = !{!13, !13, i64 0}
!13 = !{!"float", !14, i64 0}
!14 = !{!"omnipotent char", !15, i64 0}
!15 = !{!"Simple C/C++ TBAA"}
