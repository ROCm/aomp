PROGRAM test
  IMPLICIT NONE

  INTEGER, PARAMETER :: realtype=8, RT=8
  INTEGER :: NIP1, NJP1, NKP1, NIP2, NJP2, NKP2
  REAL(KIND=realtype), PARAMETER :: emf=1.0e-6_RT
  INTEGER :: i,j,k
  REAL (KIND=realtype), DIMENSION(:,:,:), POinter, COntiguous   :: f_x,f_y,f_z,f
  REAL (KIND=realtype), DIMENSION(:,:,:,:), POinter, COntiguous   :: fluxx,fluxy,fluxz
  REAL (KIND=realtype), DIMENSION(:,:,:), POinter, COntiguous   :: fc_x,fc_y,fc_z
  REAL (KIND=realtype), DIMENSION(:,:,:), POinter, COntiguous   :: ufl,vfl,wfl

  NIP1 = 513;  NJP1 = 513;  NKP1 = 513

  NIP2 = 514;  NJP2 = 514;  NKP2 = 514

  ALLOCATE(ufl(-2:NIP1, -1:NJP1, -1:NKP1))
  ALLOCATE(vfl(-1:NIP1, -2:NJP1, -1:NKP1))
  ALLOCATE(wfl(-1:NIP1, -1:NJP1, -2:NKP1))
  ALLOCATE(f(-1:NIP2, -1:NJP2, -1:NKP2))

  ALLOCATE(fluxx(-1:NIP1, -1:NJP1, -1:NKP1,3))
  ALLOCATE(fluxy(-1:NIP1, -1:NJP1, -1:NKP1,3))
  ALLOCATE(fluxz(-1:NIP1, -1:NJP1, -1:NKP1,3))

  ALLOCATE(fc_x(-1:NIP1, -1:NJP1, -1:NKP1))
  ALLOCATE(fc_y(-1:NIP1, -1:NJP1, -1:NKP1))
  ALLOCATE(fc_z(-1:NIP1, -1:NJP1, -1:NKP1))

  !if you comment any of the lines which fills one of the arrays, the kernel compiles, it also only crashes with -O0

  !$OMP TARGET TEAMS DISTRIBUTE PARALLEL DO COLLAPSE(3) DEFAULT(NONE) &
  !$OMP SHARED(emf,NIP1, NJP1, NKP1,f,fluxx,fluxy,fluxz,ufl,vfl,wfl,f_x,f_y,f_z,fc_x,fc_y,fc_z) &
  !$OMP                        PRIVATE(k,j,i)
  DO k=-1,NKP1
    DO j=-1,NJP1
      DO i=-1,NIP1
        IF(ABS(ufl(i,j,k))  <  emf) THEN
          f_x(i,j,k)=0.5_RT*(f(i,j,k)+f(i+1,j,k))
        ELSE
          f_x(i,j,k)=MIN(1.0_RT,ABS(fluxx(i,j,k,1)/ufl(i,j,k)))
        END IF

        IF(ABS(vfl(i,j,k))  <  emf) THEN
          f_y(i,j,k)=0.5_RT*(f(i,j,k)+f(i,j+1,k))
        ELSE
          f_y(i,j,k)=MIN(1.0_RT,ABS(fluxy(i,j,k,1)/vfl(i,j,k)))
        END IF
        IF(ABS(wfl(i,j,k))  <  emf) THEN
          f_z(i,j,k)=0.5_RT*(f(i,j,k)+f(i,j,k+1))
        ELSE
          f_z(i,j,k)=MIN(1.0_RT,ABS(fluxz(i,j,k,1)/wfl(i,j,k)))
        END IF
        fc_x(i,j,k)=1.0_RT-f_x(i,j,k)
        fc_y(i,j,k)=1.0_RT-f_y(i,j,k)
        fc_z(i,j,k)=1.0_RT-f_z(i,j,k)
      END DO
    END DO
  END DO
  !$OMP END TARGET TEAMS DISTRIBUTE PARALLEL DO
END PROGRAM
