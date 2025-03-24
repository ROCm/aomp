!************************************
MODULE testmodule

IMPLICIT NONE

PUBLIC :: outerprod

CONTAINS

FUNCTION outerprod(a,b)
!$OMP DECLARE TARGET
REAL(KIND=8), DIMENSION(:), INTENT(IN) :: a,b
REAL(KIND=8), DIMENSION(SIZE(a),SIZE(b)) :: outerprod
outerprod = SPREAD(a,DIM=2,ncopies=SIZE(b)) * &
  SPREAD(b,DIM=1,ncopies=SIZE(a))
END FUNCTION outerprod

END MODULE testmodule

PROGRAM test_spread
USE testmodule

END PROGRAM
!************************************
