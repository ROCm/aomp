! Copyright © Advanced Micro Devices, Inc., or its affiliates.
!
! SPDX-License-Identifier:  MIT

program prog_a
    implicit none
    type :: dtype_a
      real(4) :: var_a
      integer(4) :: var_b(3,3,3)
      real(4) :: var_c
      integer(4) :: var_d(3,3,3)
      real(4) :: var_e
    end type dtype_a

    type(dtype_a) :: var_f
    integer :: var_g, var_h, var_i

    do var_g = 1, 3
      do var_h = 1, 3
        do var_i = 1, 3
            var_f%var_b(var_g, var_h, var_i) = 42
            var_f%var_d(var_g, var_h, var_i) = 0
        end do
       end do
    end do

  !$omp target map(tofrom:var_f%var_b(1:3, 1:3, 2:2), var_f%var_d(1:3, 1:3, 1:3))
    do var_h = 1, 3
      do var_i = 1, 3
        var_f%var_d(var_i, var_h, 2) = var_f%var_b(var_i, var_h, 2)
      end do
    end do
  !$omp end target

  print *, var_f%var_d

  do var_g = 1, 3
      do var_h = 1, 3
        do var_i = 1, 3
          if (var_g == 2 .and. var_f%var_d(var_i, var_h, var_g) /= 42) then
            print *, "======= FORTRAN Test Failed! ======="
            stop 1
          else if (var_g /= 2 .and. var_f%var_d(var_i, var_h, var_g) /= 0) then
            print *, "======= FORTRAN Test Failed! ======="
            stop 1
          end if
        end do
      end do
  end do

  print*, "======= FORTRAN Test passed! ======="
end program prog_a
