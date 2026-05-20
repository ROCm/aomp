! Copyright © Advanced Micro Devices, Inc., or its affiliates.
!
! SPDX-License-Identifier:  MIT

program prog_a
type :: dtype_a
real(4) :: var_a
integer, allocatable :: var_b 
integer(4) :: var_c(10)
integer, allocatable :: var_d(:)
integer(4) :: var_e
end type dtype_a

type :: dtype_b
real(4) :: var_f
integer, allocatable :: var_g 
integer(4) :: var_h(10)
type(dtype_a) :: var_i
real(4) :: var_j
integer, allocatable :: var_k(:)
integer(4) :: var_l
end type dtype_b

type(dtype_b) :: var_m

allocate(var_m%var_i%var_b)

!$omp target map(tofrom: var_m%var_i%var_b)
    var_m%var_i%var_b = 50
!$omp end target

print *, var_m%var_i%var_b

if (var_m%var_i%var_b /= 50) then
    print *, "======= FORTRAN Test Failed! ======="
    stop 1
  end if
  
 print *, "======= FORTRAN Test Passed! ======="

end program prog_a
