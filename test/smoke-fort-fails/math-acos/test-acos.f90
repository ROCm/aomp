PROGRAM test_acos
   IMPLICIT NONE
   INTEGER, PARAMETER :: n = 10
   REAL(8), DIMENSION(n) :: x, y
   INTEGER :: i

   ! Initialize the input array with values in the valid range for acos
   DO i = 1, n
      x(i) = -1.0D0 + (2.0D0*REAL(i - 1, 8)/REAL(n - 1, 8))  ! Values from -1 to 1
   END DO

   ! Compute acos(x) on the GPU
   !$omp target map(to: x) map(from: y)
   !$omp parallel do
   DO i = 1, n
      y(i) = ACOS(x(i))
   END DO
   !$omp end target

   ! Print the results
   PRINT *, "Input Array (x): ", x
   PRINT *, "Output Array (acos(x)): ", y
END PROGRAM test_acos
