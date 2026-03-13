! Copyright © Advanced Micro Devices, Inc., or its affiliates.
!
! SPDX-License-Identifier:  MIT

program prog_a
    implicit none
    common /cblock_a/ var_a, var_b, var_c
    integer :: var_a = 0, var_b = 0, var_c = 0

    call sub_a

  !$omp target map(tofrom: var_a, var_b, var_c)
    var_c = var_c * 10
    var_b = var_b * 10
    var_a = var_a * 10
 !$omp end target

 call sub_b

  print *, "var_a after target = ", var_a
  print *, "var_b after target = ", var_b
  print *, "var_c after target = ", var_c

  if (var_a /= 310) then
      print*, "======= FORTRAN Test Failed! ======="
      stop 1
  end if

  if (var_b /= 215) then
    print*, "======= FORTRAN Test Failed! ======="
    stop 1
  end if

  if (var_c /= 420) then
    print*, "======= FORTRAN Test Failed! ======="
    stop 1
  end if

   print*, "======= FORTRAN Test Passed! ======="
end program

subroutine sub_a
  common /cblock_a/ var_a, var_b, var_c
  integer :: var_a, var_b, var_c
!$omp target map(tofrom: var_b, var_a, var_c)
  var_c = var_c + 40
  var_b = var_b + 20
  var_a = var_a + 30
!$omp end target
end

subroutine sub_b
  common /cblock_a/ var_a, var_b, var_c
  integer :: var_a, var_b, var_c
!$omp target map(tofrom: var_b, var_c, var_a)
  var_c = var_c + 20
  var_a = var_a + 10
  var_b = var_b + 15
!$omp end target
end
