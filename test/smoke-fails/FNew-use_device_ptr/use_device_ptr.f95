program main

   use omp_lib
   use iso_c_binding
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

   !$OMP TARGET DATA USE_DEVICE_PTR(arr1)
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

   do x =1, nx
       if (arr1(x) .ne. x * 2.0) then
            print *, "Wrong initialization of arr1"
            stop 1
       end if
       if (crr1(x) .ne. arr1(x)+1.0) then
            print *, "Wrong value of crr1"
            stop 1
       end if
   end do
   deallocate(arr1)
   deallocate(crr1)
  

end

