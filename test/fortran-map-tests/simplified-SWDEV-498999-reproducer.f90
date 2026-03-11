! Copyright © Advanced Micro Devices, Inc., or its affiliates.
!
! SPDX-License-Identifier:  MIT

program prog_a
    implicit none

    type dtype_a
       integer :: var_a(10)
    end type dtype_a
    integer :: var_b
    type(dtype_a), pointer :: var_c=>null()
    type(dtype_a), pointer :: var_d
    type(dtype_a), pointer :: var_e=>null()

    integer, dimension(:), pointer :: var_f

!$omp threadprivate(var_c, var_e)

nullify(var_d)
allocate(var_d)

var_e=>var_d

var_e%var_a(:)=1

var_f=>var_e%var_a

!$omp target
   do var_b = 1, 10
     var_f(var_b) = var_b
   end do
!$omp end target

print *, var_f

do var_b = 1, 10
   if (var_f(var_b) /= var_b) then
      print*, "======= FORTRAN Test Failed! ======="
      stop 1
   end if
 end do

 print*, "======= FORTRAN Test Passed! ======="

end program prog_a
