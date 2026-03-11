! Copyright © Advanced Micro Devices, Inc., or its affiliates.
!
! SPDX-License-Identifier:  MIT


program prog_a
  type :: dtype_a
  integer(4) :: var_a = 0
  real(4) :: var_b = 0.0
  complex(4) :: var_c = (0,0)
  end type dtype_a

  type(dtype_a), allocatable :: var_d
  type(dtype_a), allocatable :: var_e
  allocate(var_d)
  allocate(var_e)

  var_d%var_a = 10
  var_d%var_b = 2.0
  var_d%var_c = (2, 10)

!$omp target map(from:var_e) map(to:var_d)
    var_e%var_a = var_d%var_a
    var_e%var_b = var_d%var_b
    var_e%var_c = var_d%var_c
!$omp end target

    print*, var_d%var_a
    print*, var_d%var_b
    write (*,*) var_d%var_c

    print*, var_e%var_a
    print*, var_e%var_b
    write (*,*)  var_e%var_c

  if (var_e%var_a /= 10) then
    print*, "======= FORTRAN Test Failed! ======="
    stop 1
  end if

  if (var_e%var_b /= 2.0) then
    print*, "======= FORTRAN Test Failed! ======="
    stop 1
  end if

  if (var_e%var_c /= (2, 10)) then
    print*, "======= FORTRAN Test Failed! ======="
    stop 1
  end if

 deallocate(var_d)
 deallocate(var_e)

 print*, "======= FORTRAN Test passed! ======="
end program prog_a
