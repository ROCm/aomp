! Copyright © Advanced Micro Devices, Inc., or its affiliates.
!
! SPDX-License-Identifier:  MIT

program prog_a
    integer, allocatable :: var_a(:)
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

    !$omp target map(tofrom: var_b, var_a)
        do var_c = 1, 10
            var_b(var_c) = var_a(var_c)
        end do
    !$omp end target

    !$omp target exit data map(from: var_a)

    !$omp target exit data map(delete: var_a)

     print *, var_b

    deallocate(var_a)

    do var_c = 1, 10
      if (var_b(var_c) /= var_c + var_c) then
          print *, "======= FORTRAN Test Failed! ======="
          stop 1
      end if
  end do

  print*, "======= FORTRAN Test passed! ======="
end program
