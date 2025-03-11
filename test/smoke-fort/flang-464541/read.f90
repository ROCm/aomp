program readtest
      IMPLICIT NONE
      CHARACTER(LEN=10) :: DIR_APP
      REAL(8) :: RDUM, RISPIN
      INTEGER :: IDUM
 
      OPEN(UNIT=12,FILE='data1.dat', &
              FORM='UNFORMATTED',STATUS='UNKNOWN')
 
      READ(12,REC=1,ERR=17421) RDUM,RISPIN ; IDUM=NINT(RDUM)
 
      write(*,*) "1."
 
17421 write(*,*) "2."      
 
end program
