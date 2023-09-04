!Crown Copyright 2012 AWE.
!
! This file is part of CloverLeaf.

MODULE my_module

CONTAINS

  SUBROUTINE my_sub(x_min,x_max,y_min,y_max,             &
                            cellx,                               &
                            celly,                               &
                            volume,                              &
                            test1,                               &
                            dt_min_val)
    IMPLICIT NONE

    INTEGER :: x_min,x_max,y_min,y_max
    REAL(KIND=8)  :: dt_min_val
    REAL(KIND=8), DIMENSION(x_min-2:x_max+2)             :: cellx
    REAL(KIND=8), DIMENSION(y_min-2:y_max+2)             :: celly
    REAL(KIND=8), DIMENSION(x_min-2:x_max+2,y_min-2:y_max+2) :: volume
    REAL(KIND=8), DIMENSION(x_min-2:x_max+3,y_min-2:y_max+3) :: test1


    INTEGER          :: j,k


!$OMP TARGET TEAMS DISTRIBUTE PARALLEL DO COLLAPSE(2) &
!$OMP REDUCTION(MIN : dt_min_val) &
!$OMP MAP(tofrom: cellx, celly,volume, test1) 
    DO k=y_min,y_max
      DO j=x_min,x_max
        dt_min_val=MIN(dt_min_val,volume(j,k))
      ENDDO
    ENDDO

END SUBROUTINE my_sub

END MODULE my_module

PROGRAM main
  USE my_module
  implicit none
    INTEGER :: k,j,x_min,x_max,y_min,y_max
    REAL(KIND=8)  :: dt_min_val
    REAL(KIND=8), DIMENSION(128)             :: cellx
    REAL(KIND=8), DIMENSION(128)             :: celly
    REAL(KIND=8), DIMENSION(128,128) :: volume
    REAL(KIND=8), DIMENSION(129,129) :: test1
    dt_min_val = 0
    DO k=1,128
       cellx(k) = k
       celly(k) = k
    ENDDO
    DO k=1,128
       DO j=1,128
          volume(j,k) = j*k
       ENDDO
    ENDDO
    DO k=1,129
       DO j=1,129
          test1(j,k) = j*k
       ENDDO
    ENDDO
    call my_sub(3,126,3,126,cellx, celly, volume, test1, dt_min_val)
    DO k=1,128
       IF (cellx(k) .ne. k) THEN
          print *, "Failed cellx"
          stop 2
       ENDIF
       IF (celly(k) .ne. k) THEN
          print *, "Failed celly"
          stop 2
       ENDIF
    ENDDO
    DO k=1,128
       DO j=1,128
          IF (volume(j,k) .ne. j*k) THEN
            print *, "Failed volume"
            stop 2
          ENDIF
       ENDDO
    ENDDO
    DO k=1,129
       DO j=1,129
          IF (test1(j,k) .ne. j*k) THEN
            print *, "Failed test1"
            stop 2
          ENDIF
       ENDDO
    ENDDO
END PROGRAM main
