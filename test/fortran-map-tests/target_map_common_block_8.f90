! Copyright © Advanced Micro Devices, Inc., or its affiliates.
!
! SPDX-License-Identifier:  MIT

program prog_a
    implicit none
    common /cblock_a/ var_a, var_b, var_c
    integer :: var_a, var_b, var_c
    !$omp declare target link(/cblock_a/)

    call sub_a

  !$omp target map(tofrom: var_b)
    var_b = var_b + var_c
  !$omp end target

  call sub_b

  print *, "var_a after target = ", var_a
  print *, "var_b after target = ", var_b
  print *, "var_c after target = ", var_c

  call sub_c

  if (var_a /= 20) then
      print*, "======= FORTRAN Test Failed! ======="
      stop 1
  end if

  if (var_b /= 100) then
    print*, "======= FORTRAN Test Failed! ======="
    stop 1
  end if

  if (var_c /= 60) then
    print*, "======= FORTRAN Test Failed! ======="
    stop 1
  end if

  print*, "======= FORTRAN Test Passed! ======="
end program

subroutine sub_a
  common /cblock_a/ var_a, var_b, var_c
  integer :: var_a, var_b, var_c

!$omp target map(tofrom: /cblock_a/)
  var_a = 10
  var_b = 20
  var_c = 30
!$omp end target

end

subroutine sub_b
  common /cblock_a/ var_a, var_b, var_c
  integer :: var_a, var_b, var_c
  integer :: var_d

!$omp target map(tofrom: var_d)
  var_d =  var_b + var_c
!$omp end target

  print *, "var_d after target = ", var_d

  if (var_d /= 80) then
    print*, "======= FORTRAN Test Failed! ======="
    stop 1
  end if
end

subroutine sub_c
  common /cblock_a/ var_a, var_b, var_c
  integer :: var_a, var_b, var_c

!$omp target map(tofrom: /cblock_a/)
  var_a = var_a + var_a
  var_b = var_b + var_b
  var_c = var_c + var_c
!$omp end target
end
