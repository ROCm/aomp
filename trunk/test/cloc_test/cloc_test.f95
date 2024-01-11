
!  File: cloc_test.f95

!  Commands to compile and test
!  echo "#pragma omp declare target" > set42.c
!  echo "extern void set42(int *r_p) { *r_p = 42; }" >> set42.c
!  echo "#pragma omp end declare target" >> set42.c
!  $TRUNK/bin/clang -O3 -fopenmp --offload-arch=gfx908 -c -o set42.o set42.c
!  $TRUNK/bin/flang-new -O3 -fopenmp --offload-arch=gfx908 cloc_test.f95 set42.o -o cloc_test 
!  ./cloc_test

program main
   use, intrinsic ::  iso_c_binding
   implicit none
   interface 
      subroutine set42(result_p) bind(c,name="set42") 
         use, intrinsic :: iso_c_binding
         type(c_ptr),value :: result_p
      end subroutine set42
   end interface
   integer:: k
   integer,target :: result_i = 0
   !$omp target parallel do map(tofrom: result_i) 
   do k = 1, 256
     CALL set42(c_loc(result_i))
   end do
   !$omp end target parallel do
   if (result_i .ne. 42 ) then
     print *, "Failed result_i = ", result_i
     stop 2
   endif
   print *, "Passed result_i = ", result_i

end program main
