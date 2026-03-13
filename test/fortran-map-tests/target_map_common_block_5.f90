! Copyright © Advanced Micro Devices, Inc., or its affiliates.
!
! SPDX-License-Identifier:  MIT

program prog_a
    implicit none
    common /cblock_a/ var_a, var_b
    integer :: var_a, var_b

    call sub_a

  !$omp target map(tofrom: var_b)
      var_b = var_b + 20
  !$omp end target

    call sub_b

      print *, "var_a after target = ", var_a
      print *, "var_b after target = ", var_b

    if (var_a /= 0) then
        print*, "======= FORTRAN Test Failed! ======="
        stop 1
    end if

    if (var_b /= 400) then
      print*, "======= FORTRAN Test Failed! ======="
      stop 1
    end if

   print*, "======= FORTRAN Test Passed! ======="
end program

subroutine sub_a
  common /cblock_a/ var_a, var_b
  integer :: var_a, var_b
!$omp target map(tofrom: var_b)
  var_b = var_b + 20
!$omp end target
end

subroutine sub_b
  common /cblock_a/ var_a, var_b
  integer :: var_a, var_b
!$omp target map(tofrom: var_b)
  var_b = var_b * 10
!$omp end target
end
