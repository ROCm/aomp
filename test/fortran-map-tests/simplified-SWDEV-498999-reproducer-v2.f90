! Copyright © Advanced Micro Devices, Inc., or its affiliates.
!
! SPDX-License-Identifier:  MIT

program prog_a
    implicit none

    type dtype_a
       integer :: var_a(1000)
    end type dtype_a
    integer :: var_b
    type(dtype_a), pointer :: var_c
    type(dtype_a), pointer :: var_d=>null()

    integer, dimension(:), pointer :: var_e

!$omp THREADPRIVATE(var_d)

nullify(var_c)
allocate(var_c)

var_d=>var_c

var_d%var_a(:)=1

var_e=>var_d%var_a

!$omp target teams distribute parallel do
  do var_b = 1, 1000
   var_e(var_b) = var_b
 end do
!$omp end target teams distribute parallel do

print *, var_e

do var_b = 1, 1000
   if (var_e(var_b) /= var_b) then
      print*, "======= FORTRAN Test Failed! ======="
      stop 1
   end if
 end do

 print*, "======= FORTRAN Test Passed! ======="

end program prog_a
