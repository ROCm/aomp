program main

   use omp_lib
   implicit none
   interface
     subroutine fortran_callable_init(a,N) bind(c)
       use iso_c_binding
       implicit none
       type(c_ptr) :: a
       integer, value :: N
     end subroutine
   end interface
   integer :: nx,x
   integer, parameter :: sp = kind(1.0_4)
   real(sp), target, allocatable :: arr1(:), crr1(:)
   nx = 16
!!!!!!!! allocate arrays !!!!!!!!

   allocate(arr1(nx))
   allocate(crr1(nx))

!!!!!!!!! Initialise arrays !!!!!!!!

   arr1(:)=0.


   !$OMP TARGET DATA MAP(tofrom:arr1) MAP(from:crr1)

   !$OMP TARGET DATA USE_DEVICE_ADDR(arr1)
   call fortran_callable_init(c_loc(arr1),nx)
   !$OMP END TARGET DATA

   !$OMP TARGET TEAMS DISTRIBUTE PARALLEL DO PRIVATE(x) &
   !$OMP NOWAIT
         do x=1,nx
            crr1(x)=arr1(x)+1.0
         end do
   !$OMP END TARGET TEAMS DISTRIBUTE PARALLEL DO
      
   !$OMP TASKWAIT

   !$OMP END TARGET DATA
   print *,arr1, crr1
   deallocate(arr1)
   deallocate(crr1)
  

end

