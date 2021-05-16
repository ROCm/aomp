MODULE temperton_fft

CONTAINS
  SUBROUTINE qpassm(  d,lot)
    IMPLICIT NONE
!$OMP DECLARE TARGET

    INTEGER(4) ::  la
    INTEGER(4) ::  lot
    REAL(4) ::  d(*)
    REAL(4) ::  sin60
    REAL(4) ::  z

    INTEGER(4) ::  l, j ,jc
    DATA  sin60/0.866025403784437/

    DO l = 1, la
          d(jc+j) = z*sin60
    END DO
    RETURN
  END SUBROUTINE qpassm
 END MODULE temperton_fft
 program main 
   print *,'hi'
 end program main

 
