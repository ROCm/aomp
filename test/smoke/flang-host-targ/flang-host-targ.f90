program add_real
      implicit none
      INTEGER, PARAMETER      :: np=10
      REAL, DIMENSION(np)   :: A, B
      INTEGER:: i, getpid

      DO i=1, np
         A(i)=-1
         B(i)=1
      END DO

      !$OMP TARGET TEAMS DISTRIBUTE PARALLEL DO
      DO i=1, np
         A(i)=3*B(i)
      END DO
      !$OMP END TARGET TEAMS DISTRIBUTE PARALLEL DO

      DO i=1, np
         IF (A(i) .NE. 3) THEN
           WRITE(*, *) "ERROR AT INDEX ", i, "EXPECT 2 BUT RECEIVED", A(i)
           call kill(getpid(),7)
         ENDIF
      END DO

end program add_real
