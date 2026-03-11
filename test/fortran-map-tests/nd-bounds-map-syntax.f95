! Copyright © Advanced Micro Devices, Inc., or its affiliates.
!
! SPDX-License-Identifier:  MIT

program prog_a
    implicit none
    integer :: var_a(3,3,3)
    integer :: var_b(3,3,3)
    integer :: var_c, var_d, var_e

    do var_c = 1, 3
      do var_d = 1, 3
        do var_e = 1, 3
            var_a(var_c, var_d, var_e) = 42
            var_b(var_c, var_d, var_e) = 0
        end do
       end do
    end do

!$omp target map(tofrom:var_a(1:3, 1:3, 2:2), var_b(1:3, 1:3, 1:3))
    do var_d = 1, 3
      do var_e = 1, 3
        var_b(var_e, var_d, 2) = var_a(var_e, var_d, 2)
      end do
    end do
!$omp end target

 print *, var_b

  do var_c = 1, 3
      do var_d = 1, 3
        do var_e = 1, 3
          if (var_c == 2 .and. var_b(var_e, var_d, var_c) /= 42) then
            print *, "======= FORTRAN Test Failed! ======="
            stop 1
          else if (var_c /= 2 .and. var_b(var_e, var_d, var_c) /= 0) then
            print *, "======= FORTRAN Test Failed! ======="
            stop 1
          end if
        end do
      end do
  end do

print*, "======= FORTRAN Test passed! ======="
end program
