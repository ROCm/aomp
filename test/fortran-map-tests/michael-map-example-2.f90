program prog_a
    implicit none
    type :: dtype_a
        integer, dimension(:), allocatable :: var_a
    end type
    type(dtype_a), allocatable :: var_b
    integer, parameter :: var_c = 128
    integer :: var_d

    allocate(var_b)
    allocate(var_b%var_a(var_c))

    var_b%var_a = -42

    associate(var_e => var_b%var_a)
        !$omp target enter data map(to:var_e)

        associate(var_f => var_b%var_a)
            !$omp target enter data map(to:var_f)

            !$omp target teams distribute parallel do
            do var_d = 1, var_c
                var_f(var_d) = 42
            end do
            !$omp end target teams distribute parallel do

            !$omp target exit data map(from:var_f)
        end associate

        !$omp target exit data map(from:var_e)
    end associate

    print 100, var_b%var_a
    100 format (8I4)

    do var_d = 1, var_c
        if (var_b%var_a(var_d) /= 42) then
          print*, "======= FORTRAN Test Failed! ======="
          stop 1
       end if
    end do

    print*, "======= FORTRAN Test Passed! ======="
end program prog_a
