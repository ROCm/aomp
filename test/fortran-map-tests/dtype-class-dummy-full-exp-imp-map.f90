! Copyright © Advanced Micro Devices, Inc., or its affiliates.
!
! SPDX-License-Identifier:  MIT

module mod_a
    type :: dtype_a
        real(4) :: var_a
        real(4) :: var_b
        integer(4) :: var_c(10)
        integer(4) :: var_d
        integer(4) :: var_e(10)
    end type dtype_a
contains
subroutine sub_a(var_f)
    implicit none
    integer :: var_g
    class(dtype_a) :: var_f

    !$omp target
      do var_g = 1, 10
        var_f%var_c(var_g) = var_g
      end do
      var_f%var_d = 20
    !$omp end target
end subroutine

subroutine sub_b(var_f)
    implicit none
    integer :: var_g
    class(dtype_a) :: var_f

   !$omp target map(tofrom: var_f)
      do var_g = 1, 10
        var_f%var_e(var_g) = var_f%var_c(var_g) + 1
      end do
      var_f%var_d = 20 + var_f%var_d
   !$omp end target
end subroutine

end module mod_a

program prog_a
    use mod_a
    implicit none
    type(dtype_a) :: var_h
    integer :: var_i

    call sub_a(var_h)
    call sub_b(var_h)

    print *, var_h

    if (var_h%var_d /= 40) then
        print*, "======= FORTRAN Test Failed! ======="
        stop 1
    end if

    do var_i = 1, 10
        if (var_h%var_c(var_i) /= var_i) then
            print *, "======= FORTRAN Test Failed! ======="
            stop 1
        end if
    end do

    do var_i = 1, 10
        if (var_h%var_e(var_i) /= var_i + 1) then
            print *, "======= FORTRAN Test Failed! ======="
            stop 1
        end if
    end do

print*, "======= FORTRAN Test Passed! ======="
end program prog_a
