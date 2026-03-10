! This test works if we replace nvars with 5 in target region
MODULE rep2
        implicit none
        public :: nvars

        INTEGER :: nvars = 5
END MODULE

MODULE rep

        use rep2, ONLY:nvars
        implicit none

        public :: foo

   contains

   SUBROUTINE foo()
  implicit none
  INTEGER :: i,j,k,i1,i2
  INTEGER :: n, m
  REAL(KIND=8), DIMENSION(:,:,:), allocatable :: b

  REAL(KIND=8) :: mm(nvars), mm2(nvars)
  n=100
  m=29

  ALLOCATE(b(1:nvars,1:m,1:n))
  b=1.0_8

  !$omp target teams distribute parallel do collapse(2) private(mm,mm2)
    DO k=1,n
    DO j=1,m
      !replaced mm = 2.0_8
      DO i1 =1,nvars
         mm(i1) = 2.0_8
      END DO      
      !replaced mm2 = 1.0_8
      DO i2 =1,nvars
         mm2(i2) = 1.0_8
      END DO      
      !in the true app something else here
      DO i=1,nvars
         b(i,j,k) = b(i,j,k) + (mm(i) - mm2(i))
      END DO
    END DO
    END DO

  IF(ANY(ABS(b-2.0_8)>1.0e-9_8)) THEN
    WRITE(*,*) "failed",b(1,1,1), b(2,1,1)
    STOP 1
  ELSE
    WRITE(*,*) "success"
  END IF
  END SUBROUTINE
END MODULE

PROGRAM rep_arraysyntax
        use :: rep, ONLY:foo
        implicit none
        CALL foo()

END PROGRAM
