      SUBROUTINE BASCHK(maxg,len1,len2,len3,len4,nshell,ktype)

      integer, parameter :: mxsh=5000, mxgtot=20000
      integer, intent(in) :: nshell,ktype(nshell)
      integer, INTENT(INOUT):: maxg,len1,len2,len3,len4

      integer KANG, LMAX, NANGM, MXLEN, N
!
!     RETURN THE HIGHEST ANGULAR MOMENTUM PRESENT IN THE BASIS.
!     NOTE THAT KTYPE=1,2,3,4,5 MEANS S, P(L), D, F, G FUNCTION.
!
      KANG = 0
      DO N=1,NSHELL
          IF(KTYPE(N).GT.KANG) KANG = KTYPE(N)
      enddo
      LMAX = KANG-1

                    NANGM =  4
      IF(LMAX.EQ.2) NANGM =  6
      IF(LMAX.EQ.3) NANGM = 10
      IF(LMAX.EQ.4) NANGM = 15
      IF(LMAX.EQ.5) NANGM = 21
      IF(LMAX.EQ.6) NANGM = 28
                 MAXG = NANGM**4

      MXLEN = MAX(4, (LMAX*LMAX+3*LMAX+2)/2 )
      LEN1 = 1
      LEN2 = MXLEN
      LEN3 = MXLEN**2
      LEN4 = MXLEN**3

      RETURN
      END
!------------------------------------------------------------------
      SUBROUTINE EXPND(A,M,B,N)
      implicit none
      integer, intent(in) :: M,N
      double precision, intent(in) :: A(M)
      double precision, intent(inout) :: B(N,N)
      integer :: I,J,IJ
! expand triangular A to B(N X N)
      ij=1
      do i=1,N
        do j=1,i
           B(i,j)=A(ij)
           B(j,i)=A(ij)
           ij=ij+1
        enddo
      enddo
      end SUBROUTINE EXPND
!---------------------------------------------------------------
      SUBROUTINE CPYSQT(B,A,NA,INCA,L2)
      IMPLICIT NONE
      integer, intent(in) :: NA, INCA,L2
      double precision, intent(in) ::B(NA,NA)
      double precision, intent(inout) :: A(L2)
      double precision,PARAMETER :: SMALL=1.0D-10,ZERO=0.0D+00
      integer :: IJ,I,J
      double precision :: VAL
!!$omp declare target
!     ---- COPY SYMMETRIC, SQUARE B (N X N) TO TRIANGULAR A ----
      IJ=1
      DO I=1,NA
         DO J=1,I
            VAL = B(I,J)
            IF(ABS(VAL).LT.SMALL) VAL=ZERO
            A(IJ) = VAL
            IJ = IJ + INCA
        enddo
      enddo
      RETURN
      END
!-----------------------------------------------------------------
      SUBROUTINE CPYTSQ(A,B,NA,INCA,L2)
      IMPLICIT NONE
      integer,intent(in) :: NA,INCA,L2
      integer :: IJ,I,J
      double precision,intent(in) :: A(L2)
      double precision,intent(inout) :: B(NA,NA)
!     ---- COPY TRIANGULAR A TO SQUARE B (NA BY NA) ----
!     THE INCREMENT BETWEEN ELEMENTS OF A WILL USUALLY BE 1.
!
      IJ=1
      DO I=1,NA
         DO J=1,I
            B(I,J) = A(IJ)
            B(J,I) = A(IJ)
            IJ = IJ + INCA
         ENDDO
      ENDDO
      RETURN
      END
      
!-----------------------------------------------------------------

      SUBROUTINE VICLR(IA,INCA,N)
      implicit none
      integer, intent(in) :: INCA,N
      integer, intent(inout) :: IA(N)

      integer :: L,LA
!
!     ----- ZERO OUT AN INTEGER VECTOR -A-, USING INCREMENT -INCA- -----
!
      IF (INCA .NE. 1) GO TO 200
      DO L=1,N
         IA(L) = 0
      ENDDO
      RETURN
!
  200 CONTINUE
      LA=1-INCA
      DO L=1,N
         LA=LA+INCA
         IA(LA) = 0
      ENDDO
      RETURN
      END
!-----------------------------------------------------------------
      SUBROUTINE VCLR(A,INCA,N)
      implicit none
      double precision, parameter :: zero=0.00D+00
      integer, intent(in) :: INCA,N
      double precision, intent(inout) :: A(N)

      integer :: L,LA
!!$omp declare target
!
!     ----- ZERO OUT VECTOR -A-, USING INCREMENT -INCA- -----
!
      IF (INCA .NE. 1) GO TO 200
      DO L=1,N
         A(L) = ZERO
      ENDDO
      RETURN
!
  200 CONTINUE
      LA=1-INCA
      DO L=1,N
         LA=LA+INCA
         A(LA) = ZERO
      ENDDO
      RETURN
      END
!
!-------------------------------------------

      SUBROUTINE DCOPY(N,DX,INCX,DY,INCY)
      implicit none
      integer, intent(in) :: N, INCX, INCY
      double precision, intent(in) :: DX(INCX)
      double precision, intent(inout) :: DY(INCY)

      integer :: I, IX, IY, M, MP1
!!$omp declare target
!
!     COPIES A VECTOR.
!           DY(I) <== DX(I)
!     USES UNROLLED LOOPS FOR INCREMENTS EQUAL TO ONE.
!     JACK DONGARRA, LINPACK, 3/11/78.
!
      IF(N.LE.0) RETURN
      IF(INCX.EQ.1.AND.INCY.EQ.1)GO TO 20
!
!        CODE FOR UNEQUAL INCREMENTS OR EQUAL INCREMENTS
!          NOT EQUAL TO 1
!
      IX = 1
      IY = 1
      IF(INCX.LT.0)IX = (-N+1)*INCX + 1
      IF(INCY.LT.0)IY = (-N+1)*INCY + 1
      DO I = 1,N
        DY(IY) = DX(IX)
        IX = IX + INCX
        IY = IY + INCY
      ENDDO
      RETURN
!
!        CODE FOR BOTH INCREMENTS EQUAL TO 1
!
!
!        CLEAN-UP LOOP
!
   20 M = MOD(N,7)
      IF( M .EQ. 0 ) GO TO 40
        DO I = 1,M
          DY(I) = DX(I)
        ENDDO
      IF( N .LT. 7 ) RETURN
   40 MP1 = M + 1
        DO I = MP1,N,7
          DY(I) = DX(I)
          DY(I + 1) = DX(I + 1)
          DY(I + 2) = DX(I + 2)
          DY(I + 3) = DX(I + 3)
          DY(I + 4) = DX(I + 4)
          DY(I + 5) = DX(I + 5)
          DY(I + 6) = DX(I + 6)
        ENDDO
      RETURN
      END
!--------------------------------------------------------------------

      SUBROUTINE TRPOSQ(A,N)
      implicit none
      integer, intent(in) :: N
      double precision, intent(inout) :: A(N,N)

      integer :: I,J, JMO
      double precision :: TMP
!!$omp declare target
!
!     TRANSPOSE SQUARE MATRIX IN PLACE
!
      DO J = 2,N
         JMO = J - 1
         DO I = 1,JMO
            TMP = A(I,J)
            A(I,J) = A(J,I)
            A(J,I) = TMP
         ENDDO
      ENDDO
      RETURN
      END
!--------------------------------------------------------------------
      SUBROUTINE TRPOSE(A,B,N,M,KIND)
      implicit none
      integer, intent(in) :: N, M, KIND
      double precision, intent(inout) :: A(N,M), B(M,N)

      integer :: I,J
!
!* 14 JAN 1983 - STE * 8 MAR 1980
!*
!*    AUTHOR: S. T. ELBERT (AMES LABORATORY-USDOE)
!*
!*    PURPOSE -
!*       STORE TRANSPOSE OF N BY M MATRIX A IN MATRIX B OR A
!*            **   ****
!*
!*   ON ENTRY -
!*      A     - W.P. REAL (N,M)
!*              MATRIX TO BE TRANSPOSED
!*      N      - INTEGER
!*               ROWS OF INPUT MATRIX, COLUMNS OF OUTPUT MATRIX
!*      M      - INTEGER
!*               COLUMNS OF INPUT MATRIX, ROWS OF OUTPUT MATRIX
!*      KIND   - INTEGER
!*               IF NOT ZERO, TRANSPOSED MATRIX IS COPIED BACK INTO A
!*
!*   ON EXIT -
!*      B      - W.P. REAL (M,N)
!*               TRANSPOSED COPY OF INPUT MATRIX
!*      A (OPTIONAL) - W.P. REAL (M,N)
!*               TRANSPOSED COPY OF INPUT MATRIX
!*
!!$omp declare target
      IF(N.LE.0 .OR. M.LE.0) RETURN
      DO J=1,M
         DO I=1,N
            B(J,I) = A(I,J)
         ENDDO
      ENDDO
      IF(KIND.NE.0) CALL DCOPY(M*N,B,1,A,1)
      RETURN
      END
!-------------------------------------------------------------------------

      SUBROUTINE FMTTML_RMR(RMR,MXQT)
      implicit none
      double precision, parameter :: one=1.00D+00
      integer, intent(in) :: MXQT
      double precision, intent(inout) :: RMR(MXQT)

      integer :: M
!  RECIPROCAL FACTORS USED IN THE DOWNWARDS RECURSION FOR FM(T)
      DO M = 1, MXQT
         RMR(M) = ONE/(2*M-1)
      END DO
      END SUBROUTINE FMTTML_RMR
!------------------------------------------------------------------------
      SUBROUTINE FMTTML_TLGM(TLGM,MXQT)
      implicit none
      double precision, parameter :: one=1.00D+00
      integer,intent(in) :: MXQT
      double precision, intent(inout) :: TLGM(0:MXQT)

      double precision :: FII
      integer :: I,J, MMAX
!  (2M-1)!! FACTORS FOR LARGE-T (IN FM(T))
      MMAX=16
      DO I = 0, MMAX
         FII = ONE
         DO J = 1, 2*I-1, 2
            FII = J*FII
         END DO
         TLGM(I) = FII
      END DO
      END SUBROUTINE FMTTML_TLGM
!------------------------------------------------------------------------
      subroutine prtri(D,N)
      implicit none
      integer, intent(in) :: N
      double precision, intent(in) :: D(*)

      integer :: iw, MAXCOL, MM1, I0, IL, J0,JL, I,J
! printing triangular matrix D

      IW=6
      MAXCOL = 5
      MM1 = MAXCOL-1
      DO I0=1,N,MAXCOL
         IL = MIN(N,I0+MM1)
         WRITE(IW,9008)
         WRITE(IW,9028) (I,I=I0,IL)
         WRITE(IW,9008)
         IL = -1
         DO I=I0,N
            IL=IL+1
            J0=I0+(I*I-I)/2
            JL=J0+MIN(IL,MM1)
            WRITE(IW,9048) I,(D(J),J=J0,JL)
         ENDDO
      ENDDO
      RETURN
 9008 FORMAT(1X)
 9028 FORMAT(6X,10(4X,I4,4X))
 9048 FORMAT(I5,1X,10F12.7)
      end subroutine prtri
!-----------------------------------------------------------------------
      subroutine prvec(AMAT,N)
      IMPLICIT NONE
      integer, intent(in) :: N
      double precision, intent(in) :: AMAT(N)

      integer :: mmax,mmin,incr,i
! printing vector A

      mmax=0
      mmin=1
      incr=10
 10   continue
      mmax=mmax+incr
      if(mmax.gt.N) mmax=N
      write(6,900) (AMAT(I), I=mmin,mmax)
      mmin=mmax+1
      if(mmax.lt.N) goto 10
      return
 900  format(10(2X,F15.8))
      end subroutine prvec

!---------------------------------------------------------------------        
        SUBROUTINE CHECK_FA(FA_ref,fa_mini,l2)
        implicit none
        integer :: L2,I,Nerr
        double precision :: fa_ref(L2),fa_mini(L2),&
                            diff,threshold
        threshold = 5.0D-4

        !call FA_GAMESS(L2,fa)

        Nerr = 0
        DO I = 1, L2
           diff = abs(fa_mini(I) - fa_ref(I))
           IF(diff.GE.threshold) THEN
                !WRITE(6,FMT=100) I,FA_ref(I),diff
                Nerr = Nerr + 1
           ENDIF
        ENDDO
        if(nerr.ne.0) write(6,*) 'INCORRECT!!'
        IF(Nerr.EQ.0) write(6,*) 'FA ALL CORRECT!'

 100  FORMAT('FA(',I4,')= ',F12.9,'  INCORRECT with error ',E11.4)

        END SUBROUTINE
!----------------------------------------------------------------------        
!---- writing real matrix
      SUBROUTINE WRMATFLOAT(AMAT,N)
      IMPLICIT NONE
      integer, intent(in) :: N
      double precision, intent(in) :: AMAT(N)

      integer :: MINCOL, MAXCOL, J
      DO MINCOL = 1, N, 5
           MAXCOL=MINCOL+4
           IF(MAXCOL.GT.N) MAXCOL=N
       write(6,9048) (AMAT(J),J = MINCOL,MAXCOL)
      enddo
      RETURN
 9048 FORMAT(1X,5F17.10)
      END
!---------------------------------------------------------------------
!---- write
      SUBROUTINE WRMATINT(MAT,N)
      IMPLICIT NONE
      integer, intent(in) :: N
      integer, intent(in) :: MAT(N)

      integer :: MINCOL, MAXCOL, J

      DO MINCOL = 1, N, 10
           MAXCOL=MINCOL+9
           IF(MAXCOL.GT.N) MAXCOL=N
       write(6,9048) (MAT(J),J = MINCOL,MAXCOL)
      enddo
      RETURN
 9048 FORMAT(1X,10I7)
      END
!---------------------------------------------------------------------
!---- reading real matrix
      SUBROUTINE RDMATFLOAT(AMAT,N)
      IMPLICIT NONE
      integer, intent(in) :: N
      double precision, intent(inout) :: AMAT(N)

      integer :: MINCOL, MAXCOL, J

      DO MINCOL = 1, N, 5
           MAXCOL=MINCOL+4
           IF(MAXCOL.GT.N) MAXCOL=N
       read(5,*) (AMAT(J),J = MINCOL,MAXCOL)
       !write(6,9048) (AMAT(J),J = MINCOL,MAXCOL)
      enddo
      RETURN
 9048 FORMAT(1X,5F17.10)
      END
!--------------------------------------------------------------------
!---- reading integer matrix
      SUBROUTINE RDMATINT(MAT,N)
      IMPLICIT NONE
      integer, intent(in) :: N
      integer, intent(inout) :: MAT(N)

      integer :: MINCOL, MAXCOL, J

      DO MINCOL = 1, N, 10
           MAXCOL=MINCOL+9
           IF(MAXCOL.GT.N) MAXCOL=N
       read(*,*) (MAT(J),J = MINCOL,MAXCOL)
       !write(6,9048) (MAT(J),J = MINCOL,MAXCOL)
      enddo
      RETURN
 9048 FORMAT(1X,10I7)
      END

!--------------------------------------------------------------------
      SUBROUTINE DSCAL(N,DA,DX,INCX)
      implicit none
      integer, intent(in) :: N, INCX
      double precision, intent(inout):: DX(N)
      double precision, intent(in):: DA

      integer :: NINCX, I, M, MP1
!!$omp declare target
!
!     SCALES A VECTOR BY A CONSTANT.
!           DX(I) = DA * DX(I)
!     USES UNROLLED LOOPS FOR INCREMENT EQUAL TO ONE.
!     JACK DONGARRA, LINPACK, 3/11/78.
!
      IF(N.LE.0) RETURN
      IF(INCX.EQ.1)GO TO 20
!
!        CODE FOR INCREMENT NOT EQUAL TO 1
!
      NINCX = N*INCX
      DO I = 1,NINCX,INCX
        DX(I) = DA*DX(I)
      ENDDO
      RETURN
!
!        CODE FOR INCREMENT EQUAL TO 1
!
!
!        CLEAN-UP LOOP
!
   20 M = MOD(N,5)
      IF( M .EQ. 0 ) GO TO 40
      DO I = 1,M
        DX(I) = DA*DX(I)
      ENDDO
      IF( N .LT. 5 ) RETURN
   40 MP1 = M + 1
      DO I = MP1,N,5
        DX(I) = DA*DX(I)
        DX(I + 1) = DA*DX(I + 1)
        DX(I + 2) = DA*DX(I + 2)
        DX(I + 3) = DA*DX(I + 3)
        DX(I + 4) = DA*DX(I + 4)
      ENDDO
      RETURN
      END

