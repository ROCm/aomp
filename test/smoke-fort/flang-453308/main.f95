PROGRAM MAIN
      implicit real*8 (a-h,o-z)
      implicit integer (i-n)
      PARAMETER (NX=102,NY=64,NZ=341)

      DIMENSION AX(NX,NY,NZ) ,X(0:NX,0:NY,0:NZ)
      OM = 0.0
      IMIN = 1
      KMIN = 1

      DO K=1, NZ
        DO J=1, NY
          DO I=1, NX
             AX(I,J,K) =1.0
          ENDDO
        ENDDO
      ENDDO
      DO K=0, NZ
        DO J=0, NY
          DO I=0, NX
             X(I,J,K) = 2.0
          ENDDO
        ENDDO
      ENDDO
      CALL TEST(NX, NY, NZ, AX, X, 1, 1, OM)
END PROGRAM MAIN

SUBROUTINE TEST (NX,NY,NZ,AX,X
     &        ,IMIN,KMIN, OM)
      implicit real*8 (a-h,o-z)
      implicit integer (i-n)

      PARAMETER (NX1=102,NY1=64,NZ1=341)
      DIMENSION AX(NX1,NY1,NZ1),X(0:NX1,0:NY1,0:NZ1)

      RNORM = 0.0
!$OMP TARGET DATA MAP(RNORM) MAP(to:OM)

!$OMP TARGET TEAMS DISTRIBUTE PARALLEL DO
!$OMP&       REDUCTION(+:RNORM)
      DO K=KMIN,NZ
        DO J=1,NY
          IIMIN=IMIN+1-MOD(K+J,2)
          DO I=IIMIN,NX,2
            Xstmp   =  AX(I,J,K)*X(I-1,J,K)
            X(I,J,K) = X(I,J,K) + OM * Xstmp
            RNORM=RNORM+Xstmp**2
         ENDDO
        ENDDO
      ENDDO


!$OMP TARGET TEAMS DISTRIBUTE PARALLEL DO
!$OMP&       REDUCTION(+:RNORM)
      DO K=KMIN,NZ
        DO J=1,NY
          IIMIN=IMIN+1-MOD(K+J+1,2)
          DO I=IIMIN,NX,2
            Xstmp   =  AX(I,J,K)*X(I-1,J,K)
            X(I,J,K) = X(I,J,K) - OM * Xstmp
            RNORM=RNORM+Xstmp**2
         ENDDO
        ENDDO
      ENDDO
!$OMP END TARGET DATA
      IF (RNORM .EQ. 8904192.) THEN
       PRINT *, "======= FORTRAN Test passed! ======="
      ENDIF
 END

// CHECK: ======= FORTRAN Test passed! =======

