program main

   use omp_lib
   implicit none
 
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
   do x=1,nx
      arr1(x)=(x-1)*2.0
    end do
   !$OMP END TARGET DATA


   !$OMP TARGET TEAMS DISTRIBUTE PARALLEL DO PRIVATE(x) &
   !$OMP DEPEND(IN:var) NOWAIT
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

