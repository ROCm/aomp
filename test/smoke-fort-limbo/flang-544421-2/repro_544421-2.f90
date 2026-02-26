!
!Copyright © Advanced Micro Devices, Inc., or its affiliates.
!
!SPDX-License-Identifier:  MIT
! 
module reproducer_mod
contains
   subroutine branching_target_call(dt, dt2, dt3, switch, switch2, switch3)
      implicit none
      real(4), dimension(:), intent(inout)  :: dt
      real(4), dimension(:, :), intent(inout) :: dt2
      real(4), dimension(:, :, :), intent(inout) :: dt3
      logical, intent(in) :: switch, switch2, switch3
      integer :: dim, idx

      dim = size(dt)

      ! Large mostly irrelevant nested series of if's trying to trick
      ! the compiler into missing out on assigning to a local copy of
      ! the descriptor that'll be used inside the kernel
      if (switch) then
         !$omp target teams distribute parallel do
         do idx = 1, dim
            dt(idx) = 20
         end do
         if (switch2) then
            !$omp target teams distribute parallel do
            do idx = 1, dim
               dt(idx) = dt(idx) + 30
            end do
         else
            !$omp target teams distribute parallel do
            do idx = 1, dim
               dt(idx) = dt(idx) + 3000
            end do
         end if
         if (switch3) then
            !$omp target teams distribute parallel do
            do idx = 1, dim
               dt2(idx, 4) = dt3(idx, idx, 2) + 10
            end do
         else
            !$omp target teams distribute parallel do
            do idx = 1, dim
               dt2(idx, 4) = dt(idx) + 15
            end do
         end if
         !$omp target teams distribute parallel do
         do idx = 1, dim
            dt(idx) = dt2(idx, idx) + dt3(idx, idx, idx)
         end do
      else
         !$omp target teams distribute parallel do
         do idx = 1, dim
            dt(idx) = dt(idx) + 1000
         end do
         if (switch2) then
            !$omp target teams distribute parallel do
            do idx = 1, dim
               dt(idx) = dt2(idx, idx) + 20
            end do
            if (switch3) then
               !$omp target teams distribute parallel do
               do idx = 1, dim
                  dt(idx) = dt2(idx, idx) + dt3(idx, idx, idx)
               end do
            else
               !$omp target teams distribute parallel do
               do idx = 1, dim
                  dt3(idx, idx, idx) = dt3(idx, idx, idx) + 15
               end do
            end if
         else
            !$omp target teams distribute parallel do
            do idx = 1, dim
               dt(idx) = dt2(idx, idx) + dt3(idx, idx, idx) + 111
            end do
            if (switch3) then
               !$omp target teams distribute parallel do
               do idx = 1, dim
                  dt3(idx, idx, idx) = dt2(idx, idx) + 130
               end do
            else
               !$omp target teams distribute parallel do
               do idx = 1, dim
                  dt(idx) = dt2(idx, idx) + 120
               end do
            end if
         end if
         !$omp target teams distribute parallel do
         do idx = 1, dim
            dt(idx) = dt(idx) + 4000
         end do
      end if
   end subroutine branching_target_call
end module reproducer_mod

program reproducer
   use reproducer_mod
   implicit none
   real(4), dimension(:), allocatable :: dt
   real(4), dimension(:, :), allocatable :: dt2
   real(4), dimension(:, :, :), allocatable :: dt3
   integer :: n = 100
   integer :: i, j, k

   allocate (dt(n))
   allocate (dt2(n, n))
   allocate (dt3(n, n, n))

   dt = 1.
   dt2 = 2.
   dt3 = 3.

   ! We're less interested in the results and more interested in if we
   ! can cause a memory access violation at runtime in this test which
   ! would indicate a regression in the handling of the input arguments
   ! being mapped to targets in multiple divergent branches.
   call branching_target_call(dt, dt2, dt3, .false., .false., .false.)
   call branching_target_call(dt, dt2, dt3, .false., .true., .false.)
   call branching_target_call(dt, dt2, dt3, .false., .true., .true.)
   call branching_target_call(dt, dt2, dt3, .true., .true., .true.)
   call branching_target_call(dt, dt2, dt3, .true., .true., .false.)
   call branching_target_call(dt, dt2, dt3, .true., .false., .false.)
   call branching_target_call(dt, dt2, dt3, .true., .false., .true.)
   call branching_target_call(dt, dt2, dt3, .false., .false., .true.)

   print *, "success"
end program reproducer
