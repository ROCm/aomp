PROGRAM test_target_teams_distribute_depend
  USE iso_fortran_env
  USE omp_lib
  implicit none
  INTEGER :: errors
  INTEGER, PARAMETER :: N = 1024
  errors = 0

CONTAINS
  INTEGER FUNCTION depend_in_out()
    INTEGER :: errors_a, errors_b, x
    INTEGER, DIMENSION(N) :: a, b, c, d

    errors_a = 0
    errors_b = 0

    DO x = 1, N
       a(x) = x
       b(x) = 2 * x
       c(x) = 0
       d(x) = 0
    END DO

    !$omp target data map(to: a(1:N), b(1:N)) map(alloc: c(1:N)) map( &
    !$omp& d(1:N))
    !$omp target teams distribute nowait depend(in: c) map(alloc: &
    !$omp& a(1:N), b(1:N), c(1:N))
    DO x = 1, N
       c(x) = a(x) + b(x)
    END DO
    !$omp target teams distribute nowait depend(out: c) map(alloc: &
    !$omp& b(1:N), c(1:N), d(1:N))
    DO x = 1, N
       d(x) = c(x) + b(x)
    END DO
    !$omp taskwait
    !$omp end target data

    !$omp target data map(to: a(1:N), b(1:N)) map(alloc: c(1:N)) map( &
    !$omp from: d(1:N))
    !$omp target teams distribute nowait depend(in: c) map(alloc: &
    !$omp& a(1:N), b(1:N), c(1:N))
    DO x = 1, N
       c(x) = a(x) + b(x)
    END DO
    !$omp target teams distribute nowait depend(inout: c) map(alloc: &
    !$omp& a(1:N), c(1:N), d(1:N))
    DO x = 1, N
       d(x) = c(x) + a(x)
    END DO
    !$omp taskwait
    !$omp end target data

    depend_in_out = errors_a + errors_b
  END FUNCTION depend_in_out
END PROGRAM test_target_teams_distribute_depend
