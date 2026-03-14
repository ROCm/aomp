! Copyright © Advanced Micro Devices, Inc., or its affiliates.
!
! SPDX-License-Identifier:  MIT

module mod_a
    type :: dtype_a
            integer :: var_a, var_b
            integer,dimension(:),allocatable :: var_c
    end type
end module

program prog_a
    use mod_a

    type(dtype_a),target :: var_d
    integer,dimension(:),pointer :: var_e
    integer :: var_f

    allocate(var_d%var_c(1024))
    var_d%var_a=1
    var_d%var_b=1024

    var_e => var_d%var_c

    !$omp target enter data map(to:var_d, var_d%var_c)

    !$omp target
      do var_f = 1,1024
             var_e(var_f) = var_f
      end do
    !$omp end target

    !$omp target exit data map(from:var_d%var_c)

    write(*,*) var_d%var_c

    !$omp target exit data map(release:var_d)

    deallocate(var_d%var_c)
end program
