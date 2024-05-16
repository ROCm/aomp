! compile with: flang-new  -O3  -fopenmp --offload-arch=gfx90a -Dundef -DPERF -c matsol.f

      SUBROUTINE SOR13D (NX,NY,NZ,AX,B,CX,AY,CY,AZ,CZ,RHS,X
     &        ,OMEGA0,OMOLD,XSMIN,IDIR,ITOLD,ITmax,EPS0,IPRT,ICONVERGE
     &        ,IMIN,KMIN
     &        ,IPSLR)
      implicit real*8 (a-h,o-z)
      implicit integer (i-n)

      PARAMETER (NX1=102,NY1=64,NZ1=341,
     &           NX2=103,NY2=65,NZ2=342,NXYZMAX=342)
    
      DIMENSION  XS(NX1,NY1,NZ1),BRES(NX1,NY1,NZ1)

      DIMENSION AX(NX1,NY1,NZ1),BX(NX1,NY1,NZ1),CX(NX1,NY1,NZ1)
     &          ,AY(NX1,NY1,NZ1),BY(NX1,NY1,NZ1),CY(NX1,NY1,NZ1)
     &          ,AZ(NX1,NY1,NZ1),BZ(NX1,NY1,NZ1),CZ(NX1,NY1,NZ1)
     &          ,RHS(NX1,NY1,NZ1),X(0:NX1,0:NY1,0:NZ1)
     &          ,B(NX1,NY1,NZ1),DUM(2:NX-1,2:NY-1,2:NZ)
     &       , subd(NXYZMAX), suprad(NXYZMAX)
     &       , diag(NXYZMAX), rest(NXYZMAX)

!$OMP TARGET DATA MAP(RNORM) MAP(to:OM)

!$OMP TARGET TEAMS DISTRIBUTE PARALLEL DO
!$OMP&       REDUCTION(+:RNORM)
      DO K=KMIN,NZ
        DO J=1,NY
          IIMIN=IMIN+1-MOD(K+J,2)
          DO I=IIMIN,NX,2
            Xstmp   =  AX(I,J,K)*X(I-1,J,K)+CX(I,J,K)*X(I+1,J,K)
     &                +AY(I,J,K)*X(I,J-1,K)+CY(I,J,K)*X(I,J+1,K)
     &                +AZ(I,J,K)*X(I,J,K-1)+CZ(I,J,K)*X(I,J,K+1)
     &                +X(I,J,K)
     &                -RHS(I,J,K)
            X(I,J,K) = X(I,J,K) - OM * Xstmp
            RNORM=RNORM+Xstmp**2
         ENDDO
        ENDDO
      ENDDO

!      CALL RADBOU(X,.FALSE.)

!$OMP TARGET TEAMS DISTRIBUTE PARALLEL DO
!$OMP&       REDUCTION(+:RNORM)
      DO K=KMIN,NZ
        DO J=1,NY
          IIMIN=IMIN+1-MOD(K+J+1,2)
          DO I=IIMIN,NX,2
            Xstmp   =  AX(I,J,K)*X(I-1,J,K)+CX(I,J,K)*X(I+1,J,K)
     &                +AY(I,J,K)*X(I,J-1,K)+CY(I,J,K)*X(I,J+1,K)
     &                +AZ(I,J,K)*X(I,J,K-1)+CZ(I,J,K)*X(I,J,K+1)
     &                +X(I,J,K)
     &                -RHS(I,J,K)
            X(I,J,K) = X(I,J,K) - OM * Xstmp
            RNORM=RNORM+Xstmp**2
         ENDDO
        ENDDO
      ENDDO
!$OMP END TARGET DATA
      END
