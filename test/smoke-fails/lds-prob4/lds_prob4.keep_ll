target triple = "amdgcn-amd-amdhsa"
  
define i64 @f(i64 %x)  {
entry:
        br i1 undef, label %return, label %if.end


        if.end:
        %call = call i64 @f(i64 undef)
        unreachable


        return:
        ret i64 1
}


define i8* @unrelated() {
entry:
 ret i8* bitcast (i64 (i64)* @f to i8*)
}


declare amdgpu_kernel void @has_a_kernel(i32)






