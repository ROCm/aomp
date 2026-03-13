! Copyright © Advanced Micro Devices, Inc., or its affiliates.
!
! SPDX-License-Identifier:  MIT

subroutine sub_a(var_a, var_b, var_c)
    implicit none
    real, intent(out) :: var_c
    real(4), dimension(var_a,var_b) :: var_d
    integer :: var_a,var_b
!$omp target enter data map(alloc:var_d)

!$omp target
      var_d(2, 2) = 10
!$omp end target

!$omp target update from(var_d)

   var_c = var_d(2, 2)

!$omp target exit data map(delete:var_d)
end subroutine sub_a

subroutine sub_b(var_a, var_b)
    implicit none
    real, intent(inout) :: var_b
    integer :: var_a
    real(4), dimension(var_a) :: var_c

!$omp target enter data map(to:var_c)

!$omp target
    var_c(8) = 20
!$omp end target

!$omp target update from(var_c)

var_b = var_b + var_c(8)

end subroutine sub_b

program prog_a
    implicit none
    integer :: var_a = 10
    integer :: var_b = 10, var_c = 10
    real :: var_d = 0

    call sub_a(var_b, var_c,  var_d)
    call sub_b(var_a, var_d)

    print *, var_d

    if (var_d /= 30) then
        print *, "======= FORTRAN Test Failed! ======="
        stop 1
      end if

     print *, "======= FORTRAN Test Passed! ======="
end program prog_a
