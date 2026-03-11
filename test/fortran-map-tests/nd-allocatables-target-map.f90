! Copyright © Advanced Micro Devices, Inc., or its affiliates.
!
! SPDX-License-Identifier:  MIT

program prog_a
  implicit none
  integer,  allocatable :: var_a(:,:,:)
  integer :: var_b = 1
  integer :: var_c, var_d, var_e

  allocate(var_a(3,3,3))

!$omp target map(tofrom: var_a)
    do var_c = 1, 3
      do var_d = 1, 3
        do var_e = 1, 3
           var_a(var_e, var_d, var_c) = var_b
           var_b = var_b + 1
        end do
      end do
    end do
!$omp end target

    print *, var_a

    do var_c = 1, 3
        do var_d = 1, 3
          do var_e = 1, 3
            if (var_a(var_e, var_d, var_c) /= var_b) then
                print*, "======= FORTRAN Test Failed! ======="
                stop 1
            end if
           var_b = var_b + 1
          end do
        end do
      end do

    print*, "======= FORTRAN Test Passed! ======="
end program
