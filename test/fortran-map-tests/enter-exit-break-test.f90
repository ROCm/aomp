! Copyright © Advanced Micro Devices, Inc., or its affiliates.
!
! SPDX-License-Identifier:  MIT

module mod_a
integer, allocatable :: var_a(:)
contains
subroutine sub_a(var_b)
    implicit none
    integer, intent(inout) :: var_b(:)
    integer :: var_c

    !$omp target
      do var_c=1,10
        var_b(var_c) = var_a(var_c)
      end do
    !$omp end target
end subroutine sub_a
end module mod_a

program prog_a
    use mod_a
    implicit none
    integer :: var_b(10)
    integer :: var_c

    allocate(var_a(10))

    do var_c = 1, 10
        var_a(var_c) = var_c + var_c
    end do

    !$omp target enter data map(to: var_a)
    
    do var_c = 1, 10
        var_a(var_c) = var_c
    end do

    call sub_a(var_b)

    print *, var_b

    do var_c = 1, 10
        if (var_b(var_c) /= var_c + var_c) then
            print *, "======= FORTRAN Test Failed! ======="
            stop 1
        end if
    end do

  print*, "======= FORTRAN Test passed! ======="

end program
