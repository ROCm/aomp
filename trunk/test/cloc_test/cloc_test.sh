#!/bin/bash
#  File: cloc_test.sh
#
# create sources target_atomic_sum.c and cloc_test.f95
echo "Creating target_atomic_sum.c"
/bin/cat >target_atomic_sum.c <<"EOF"
#pragma omp declare target
extern void target_atomic_sum(int val0, int *r_p) {
  #pragma omp atomic
  *r_p += val0;
}
#pragma omp end declare target
EOF
echo "Creating cloc_test.f95"
/bin/cat >cloc_test.f95 <<"EOF"
program main
   use, intrinsic ::  iso_c_binding
   implicit none
   interface
      subroutine target_atomic_sum(tls_val,result_p) bind(c,name="target_atomic_sum")
         use, intrinsic :: iso_c_binding
         integer,value :: tls_val
         type(c_ptr),value :: result_p
      end subroutine target_atomic_sum
   end interface
   integer:: k, validate
   integer:: sz = 1025
   integer:: tls_val
   integer,target :: result_i = 1
   !$omp target teams distribute parallel do map(tofrom: result_i)
   do k = 1, sz
     tls_val = k
     CALL target_atomic_sum(tls_val,c_loc(result_i))
   end do
   !$omp end target teams distribute parallel do

   validate = ((( sz + 1 ) * sz) / 2 ) + 1
   if (result_i .ne. validate ) then
     print *, "Failed sz=", sz, " result_i=", result_i, " validate=" , validate
     stop 2
   endif
   print *, "Passed sz=", sz, " result_i=", result_i, " validate=" , validate
end program main
EOF

TRUNK=${TRUNK:-$HOME/rocm/trunk}
FLANG=${FLANG:-flang-new}
echo $TRUNK/bin/clang -O3 -fopenmp --offload-arch=gfx908 -c -o target_atomic_sum.o target_atomic_sum.c
$TRUNK/bin/clang -O3 -fopenmp --offload-arch=gfx908 -c -o target_atomic_sum.o target_atomic_sum.c
echo "$TRUNK/bin/$FLANG -v -save-temps -O3 -fopenmp --offload-arch=gfx908 cloc_test.f95 target_atomic_sum.o -o cloc_test 2>compile.stderr"
$TRUNK/bin/$FLANG -v -save-temps -O3 -fopenmp --offload-arch=gfx908 cloc_test.f95 target_atomic_sum.o -o cloc_test 2>compile.stderr
echo "LIBOMPTARGET_DEBUG=1 ./cloc_test 2>run.stderr"
LIBOMPTARGET_DEBUG=1 ./cloc_test 2>run.stderr
