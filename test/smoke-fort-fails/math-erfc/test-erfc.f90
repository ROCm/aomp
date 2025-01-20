! build error:
! ld.lld: error: undefined symbol: erfc
PROGRAM test_erfc
   IMPLICIT NONE
   INTEGER, PARAMETER :: n = 10
   REAL(8), DIMENSION(n) :: x, y
   INTEGER :: i

   ! Initialize the array
   x = [(REAL(i, 8), i=1, n)]

   ! Compute erfc on GPU
   !$omp target map(to: x) map(from: y)
   !$omp parallel do
   DO i = 1, n
      y(i) = erfc(x(i))
   END DO
   !$omp end target

   ! Print the result
   PRINT *, "Input Array: ", x
   PRINT *, "Output Array (erfc): ", y
END PROGRAM test_erfc
