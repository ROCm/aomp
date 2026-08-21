MODULE foo
  USE iso_c_binding
  USE omp_lib
  IMPLICIT NONE
  PRIVATE
  PUBLIC :: bar_device_addr, print_ptr, bar

  INTERFACE
    SUBROUTINE bar(x, n) BIND(C, name="bar_GPU")
      USE iso_c_binding
      TYPE(C_PTR),    VALUE, INTENT(IN)    :: x
      INTEGER(C_INT), VALUE, INTENT(IN)    :: n
    END SUBROUTINE
    SUBROUTINE print_ptr(x) BIND(C, name="print_ptr")
      USE iso_c_binding
      TYPE(C_PTR),    VALUE, INTENT(IN)    :: x
    END SUBROUTINE

  END INTERFACE

CONTAINS

  SUBROUTINE bar_device_addr(x, n)
    INTEGER, TARGET, INTENT(IN)    :: x(:)
    INTEGER(C_INT), INTENT(IN)     :: n
    !$omp target data use_device_addr (x)
    CALL print_ptr(c_loc(x))
    CALL bar(c_loc(x), n)
    !$omp end target data
  END SUBROUTINE

END MODULE foo

PROGRAM test_ptr
  USE iso_c_binding
  USE, intrinsic :: iso_fortran_env, only: error_unit
  USE omp_lib
  USE foo
  IMPLICIT NONE

  INTEGER, ALLOCATABLE, TARGET :: x(:)
  INTEGER(C_INT) :: i, n
  n = 1000
  ALLOCATE(x(n))
  x = 1
  CALL print_ptr(c_loc(x))
  !$omp target enter data map(to: x)
     CALL print_ptr(c_loc(x))
  !$omp target data use_device_addr (x)
    CALL print_ptr(c_loc(x))
    CALL bar(c_loc(x), n)
  !$omp end target data
  CALL bar_device_addr(x,n)
  !$omp target exit data map(from: x)
  CALL print_ptr(c_loc(x))
  DO i = 1,n
     IF (x(i) .ne. 3) then
       PRINT *, "Bad result for use_device_addr!"
       STOP 1
     ENDIF
  END DO
  DEALLOCATE(x)
  write(error_unit, *) 'Success'
END PROGRAM test_ptr

