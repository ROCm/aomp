! Copyright © Advanced Micro Devices, Inc., or its affiliates.
!
! SPDX-License-Identifier:  MIT

module mod_a
    contains

 subroutine sub_a(var_a,var_b,var_c,var_d)
     implicit none
     integer :: var_e
     integer, dimension(*), intent(out) :: var_a(:)
     integer, dimension(*), intent(in) :: var_b(:), var_c(:)
     integer, intent(in) :: var_d

     !$omp target map(to: var_b, var_c) map(from: var_a)
         do var_e=1,var_d
             var_a(var_e) = var_b(var_e) * var_c(var_e)
         end do
     !$omp end target
  end subroutine
 end module mod_a

 program prog_a
    use mod_a
    implicit none
    integer :: var_a = 5
    integer, allocatable :: var_b(:)
    integer, allocatable :: var_c(:)
    integer, allocatable :: var_d(:)

    allocate(var_b(5))
    allocate(var_c(5))
    allocate(var_d(5))

    var_b = (/0, 0, 0, 0, 0/)
    var_c = (/1, 2, 3, 4, 5/)
    var_d = (/6, 7, 8, 9, 10/)

    call sub_a(var_b, var_c, var_d, var_a)

    print *, var_b

    if (var_b(1) /= 6) then
       print*, "======= FORTRAN Test Failed! ======="
       stop 1
    end if

    if (var_b(2) /= 14) then
       print*, "======= FORTRAN Test Failed! ======="
       stop 1
    end if

    if (var_b(3) /= 24) then
       print*, "======= FORTRAN Test Failed! ======="
       stop 1
    end if

    if (var_b(4) /= 36) then
       print*, "======= FORTRAN Test Failed! ======="
       stop 1
    end if

    if (var_b(5) /= 50) then
       print*, "======= FORTRAN Test Failed! ======="
       stop 1
    end if

    print*, "======= FORTRAN Test Passed! ======="
 end program
