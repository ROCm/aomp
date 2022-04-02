PROGRAM test_target_teams_distribute_depend
  USE iso_fortran_env
  USE omp_lib
  implicit none
  INTEGER :: errors
  INTEGER, parameter :: N = 1024
  errors = depend_array_section()
  if (errors .ne. 0) stop 2

CONTAINS
  INTEGER FUNCTION depend_array_section()
    INTEGER :: errors, x
    INTEGER,DIMENSION(N) :: a, b, c, d

    DO x = 1, N
       a(x) = x
       b(x) = 2 * x
       c(x) = 0
       d(x) = 0
    END DO

    !$omp target data map(to: a(1:N), b(1:N)) map(alloc: c(1:N)) map( &
    !$omp& from: d(1:N))
    !$omp target teams distribute nowait depend(out: c(1:N)) map(alloc: &
    !$omp& a(1:N), b(1:N), c(1:N))
    DO x = 1, N
       c(x) = a(x) + b(x)
    END DO
    !$omp target teams distribute nowait depend(out: c(1:N)) map(alloc: &
    !$omp& b(1:N), c(1:N), d(1:N))
    DO x = 1, N
       d(x) = c(x) + b(x)
    END DO
    !$omp taskwait
    !$omp end target data

    errors = 0
    do x = 1,10
    if (d(x) .eq. 0) errors = 1
    end do
    depend_array_section = errors
  END FUNCTION depend_array_section
END PROGRAM test_target_teams_distribute_depend 


