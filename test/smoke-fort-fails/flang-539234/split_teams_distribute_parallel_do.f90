!
!Copyright © Advanced Micro Devices, Inc., or its affiliates.
!
!SPDX-License-Identifier:  MIT
! 
! compile with:
!
! amdflang -O2 -fopenmp --offload-arch=gfx90a -o split_teams_distribute_parallel_do split_teams_distribute_parallel_do.f90

module foo_mod
   implicit none
contains
   subroutine foo(ib,val,arr)
      implicit none
      integer, intent(in) :: ib, val
      integer, dimension(:), intent(inout) :: arr
      integer :: i
      !$omp declare target(foo)

      !$omp parallel do
      do i=1,ib
         arr(i)=arr(i)+val
      end do
   end subroutine foo
end module foo_mod

program hierarchical
   use foo_mod
   implicit none
   integer :: i, j, val
   integer :: a(10,10)

   a=1
   val=2
   !$omp target enter data map(to:a)

   !$omp target teams distribute
   do j=1,10
      call foo(10,val,a(:,j))
   end do

   !$omp target update from(a)
   do j=1,10
      do i=1,10
         if (a(i,j)/=3) then
            print *, 'Error for indexes', i, j
            print *, 'Value is', a(i,j)
            stop 1
         end if
      end do
   end do
   print *, 'Success!!!'
end program hierarchical
