! Copyright © Advanced Micro Devices, Inc., or its affiliates.
!
! SPDX-License-Identifier:  MIT

program prog_a
    implicit none
    type :: dtype_a
      real(4) :: var_a
      integer(4) :: var_b(10)
      integer(4) :: var_c
    end type dtype_a

    type :: dtype_b
      real(4) :: var_e
      integer, allocatable :: var_f
      integer(4) :: var_g(10)
      real(4) :: var_h
      integer, allocatable :: var_i(:,:,:)
      integer(4) :: var_j
      type(dtype_a) :: var_k
    end type dtype_b

    type(dtype_b) :: var_l
    integer :: var_m(3,3,3)
    integer :: var_n, var_o, var_p

    allocate(var_l%var_i(3,3,3))
    allocate(var_l%var_f)

    do var_n = 1, 3
        do var_o = 1, 3
          do var_p = 1, 3
              var_m(var_n, var_o, var_p) = 42
              var_l%var_i(var_n, var_o, var_p) = 0
          end do
         end do
      end do

!$omp target map(tofrom: var_l%var_i(1:3, 1:3, 2:2)) map(to: var_m(1:3, 1:3, 1:3))
    do var_o = 1, 3
        do var_p = 1, 3
            var_l%var_i(var_p, var_o, 2) = var_m(var_p, var_o, 2)
        end do
      end do
!$omp end target

    print *, var_l%var_i

    do var_n = 1, 3
        do var_o = 1, 3
          do var_p = 1, 3
            if (var_n == 2 .and. var_l%var_i(var_p, var_o, var_n) /= 42) then
              print *, "======= FORTRAN Test Failed! ======="
              stop 1
            else if (var_n /= 2 .and. var_l%var_i(var_p, var_o, var_n) /= 0) then
              print *, "======= FORTRAN Test Failed! ======="
              stop 1
            end if
          end do
        end do
    end do

    print *, "======= FORTRAN Test Passed! ======="
end program prog_a
