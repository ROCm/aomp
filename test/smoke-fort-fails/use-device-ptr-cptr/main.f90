MODULE foo
  USE iso_c_binding
  USE omp_lib
  IMPLICIT NONE
  PRIVATE
  PUBLIC :: bar_device_ptr, bar_device_addr

  INTERFACE
    SUBROUTINE bar(x, y, z, n) BIND(C, name="bar_GPU")
      USE iso_c_binding
      TYPE(C_PTR),    VALUE, INTENT(IN)    :: x, y, z
      INTEGER(C_INT), VALUE, INTENT(IN)    :: n
    END SUBROUTINE

  END INTERFACE

CONTAINS

  SUBROUTINE bar_device_addr(x,y,z,n)
    INTEGER, TARGET, INTENT(IN)    :: x(:), y(:)
    INTEGER, TARGET, INTENT(INOUT) :: z(:)
    INTEGER(C_INT), INTENT(IN)     :: n
    !$omp target data use_device_addr (x, y, z)
    CALL bar(c_loc(x), c_loc(y), c_loc(z), n)
    !$omp end target data
  END SUBROUTINE

  SUBROUTINE bar_device_ptr(x,y,z,n)
    INTEGER, TARGET, INTENT(IN)    :: x(:), y(:)
    INTEGER, TARGET, INTENT(INOUT) :: z(:)
    INTEGER(C_INT), INTENT(IN)     :: n
    TYPE(C_PTR)                    :: x_ptr, y_ptr, z_ptr

    x_ptr = c_loc(x)
    y_ptr = c_loc(y)
    z_ptr = c_loc(z)
    !$omp target data use_device_ptr (x_ptr, y_ptr, z_ptr)
    CALL bar(x_ptr, y_ptr, z_ptr, n)
    !$omp end target data
  END SUBROUTINE

END MODULE foo

PROGRAM test_ptr
  USE iso_c_binding
  USE omp_lib
  USE foo
  IMPLICIT NONE

  INTEGER, ALLOCATABLE, TARGET :: x(:), y(:), z(:)
  INTEGER, ALLOCATABLE, TARGET :: x1(:), y1(:), z1(:)
  INTEGER(C_INT) :: i, n
  n = 1000
  ALLOCATE(x(n), y(n), z(n))
  ALLOCATE(x1(n), y1(n), z1(n))
  z = 0
  z1 = 0
  x = 1
  y = 2
  x1 = 1
  y1 = 2
  i = 1
  !$omp target enter data map(to: x,y,z,x1,y1,z1)

  CALL bar_device_addr(x,y,z,n)
  CALL bar_device_ptr(x1,y1,z1,n)
  !$omp target exit data map(from: x,y,z,x1,y1,z1)
  DO i = 1,n
     IF (z(i) .ne. 3) then
       PRINT *, "Bad result for use_device_addr!"
       STOP 1
     ENDIF
     IF (z1(i) .ne. 3) then
       PRINT *, "Bad result for use_device_ptr!"
       STOP 1
     ENDIF
  END DO
  DEALLOCATE(x,y,z,x1,y1,z1)
  PRINT *, "Success"
END PROGRAM test_ptr

