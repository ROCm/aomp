!
!Copyright © Advanced Micro Devices, Inc., or its affiliates.
!
!SPDX-License-Identifier:  MIT
! 
module reproducer_mod
contains
   subroutine branching_target_call(dt, switch)
      implicit none
      real(4), dimension(:), intent(inout) :: dt
      logical, intent(in) :: switch
      integer :: dim, idx

      dim = size(dt)

      if (switch) then

!$omp target teams distribute parallel do
         do idx = 1, dim
            dt(idx) = 20
         end do

      else

!$omp target teams distribute parallel do
         do idx = 1, dim
            dt(idx) = 30
         end do
      end if

   end subroutine branching_target_call
end module reproducer_mod

program reproducer
   use reproducer_mod
   implicit none
   real(4), dimension(:), allocatable :: dt
   integer :: n = 21312
   integer :: i

   allocate (dt(n))

   call branching_target_call(dt, .FALSE.)

   do i = 1, n
      if (dt(i) /= 30) then
         print *, "failed"
      end if
   end do

   call branching_target_call(dt, .TRUE.)

   do i = 1, n
      if (dt(i) /= 20) then
         print *, "failed"
      end if
   end do

   print *, "success"
end program reproducer
