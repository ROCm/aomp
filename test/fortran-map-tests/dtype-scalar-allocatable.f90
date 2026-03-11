! Copyright © Advanced Micro Devices, Inc., or its affiliates.
!
! SPDX-License-Identifier:  MIT

program prog_a

type :: dtype_a
real(4) :: var_a
integer, allocatable :: var_b 
integer(4) :: var_c(10)
real(4) :: var_d
integer, allocatable :: var_e(:)
integer(4) :: var_f
end type dtype_a

type(dtype_a) :: var_g

allocate(var_g%var_b) 

!$omp target map(tofrom: var_g%var_b)
    var_g%var_b = 50
!$omp end target

print *, var_g%var_b

if (var_g%var_b /= 50) then
    print *, "======= FORTRAN Test Failed! ======="
    stop 1
  end if
  
 print *, "======= FORTRAN Test Passed! ======="

end program prog_a
