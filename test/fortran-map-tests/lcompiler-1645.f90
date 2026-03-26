program prog_a
  implicit none
    type :: dtype_a
        real, allocatable :: var_a
    end type dtype_a

    type :: dtype_b
        integer(4) :: var_b
        type(dtype_a) :: var_c
    end type dtype_b

    type :: dtype_c
        integer(4) :: var_d = 0
        real(4) :: var_e = 0.0
        complex(4) :: var_f = (0,0)
        real(4) :: var_g = 1.0
        type(dtype_b) :: var_h
    end type dtype_c

    type(dtype_c) :: var_i

    allocate(var_i%var_h%var_c%var_a)

    !$OMP TARGET MAP(TOFROM: var_i%var_d, var_i%var_h%var_c%var_a)
        var_i%var_d = 10
        var_i%var_h%var_c%var_a = 20
    !$OMP END TARGET

    print *, var_i%var_d
    print *, var_i%var_h%var_c%var_a

    if (var_i%var_d /= 10) then
        print*, "======= FORTRAN Test Failed! ======="
        stop 1
    end if

    if (var_i%var_h%var_c%var_a /= 20) then
        print*, "======= FORTRAN Test Failed! ======="
        stop 1
    end if

    print*, "======= FORTRAN Test Passed! ======="
end program
