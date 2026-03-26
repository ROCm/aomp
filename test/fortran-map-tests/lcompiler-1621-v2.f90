program prog_a
    implicit none
    common/cblock_a/var_a
    real(kind=8), dimension(100) :: var_a
   !$omp declare target(/cblock_a/)
    integer var_b

    var_a = 1.0d0
    !$omp target enter data map(always, to: var_a)

    !$omp target teams distribute parallel do
        do var_b = 1, 100
            var_a(var_b) = var_a(var_b) * 2.0d0
        enddo
    !$omp end target teams distribute parallel do

    !$omp target update from(var_a)

    call verification(var_a, 2.0d0)

    !$omp parallel do
        do var_b = 1, 100
            var_a(var_b) = var_a(var_b) * 2.0d0
        enddo
    !$omp end parallel do

    call verification(var_a, 4.0d0)

    print*, "======= FORTRAN Test Passed! ======="
end program

subroutine verification(var_c, var_d)
    real(kind=8), dimension(100) :: var_c
    real(kind=8) :: var_d
    integer :: var_e

    write(*,*) var_c

    do var_e = 1, 100
      if (var_c(var_e) /= var_d) then
        print*, "======= FORTRAN Test Failed! ======="
        stop 1
      end if
    enddo
end subroutine