 module mod_dirfck
 contains
      subroutine dirfck(FA2d,Da2d,GHONDO, KTYPE,KMIN,KMAX,KLOC, &
                 ishell,jshell,kshell,lshell,nshell,minshell, &
                 lstri,lstrj,lstrk,lstrl,L1,maxg)
      integer,parameter :: MXSH=5000 
      integer :: ishell,jshell,kshell,lshell,nshell,minshell
      integer :: MINI,MINJ,MINK,MINL,MAXI,MAXJ,MAXK,MAXL, &
                 LOCI,LOCJ,LOCK,LOCL,NINTN
      integer :: I,J,K,L,I1,J1,K1,LL1,I_INDEX,IJ_INDEX,IJK_INDEX,IJKL_INDEX
      integer :: lstri,lstrj,lstrk,lstrl,len1,len2,len3,len4
      integer :: L1,L2,maxg,Nerr
      double precision :: xval1,xval4,val,val1,val4,diff 
      integer,dimension (MXSH) :: KTYPE,KMIN,KMAX,KLOC
      double precision :: FA2d(l1,l1),Da2d(l1,l1),GHONDO(maxg)

      double precision,parameter :: F04=4.0D+00,cutoff = 1.00E-010,&
                                    HFSCAL=1.0D+00,CSCALT=1.0D+00,&
                                  one_eight=0.125D+00,threshold=1.0D-10,&
                                    zero=0.0D+00
!$omp declare target
      MINI = KMIN(ishell)
      MINJ = KMIN(jshell)
      MINK = KMIN(kshell)
      MINL = KMIN(lshell)
      MAXI = KMAX(ishell)
      MAXJ = KMAX(jshell)
      MAXK = KMAX(kshell)
      MAXL = KMAX(lshell)
      LOCI = KLOC(ishell)-MINI
      LOCJ = KLOC(jshell)-MINJ
      LOCK = KLOC(kshell)-MINK
      LOCL = KLOC(lshell)-MINL
      XVAL1 = HFSCAL
      XVAL4 = F04*CSCALT

      DO I = MINI,MAXI
         DO J = MINJ,MAXJ
            DO K =  MINK,MAXK
               DO L=MINL,MAXL

                  IJ_INDEX = (J-MINJ)*lstrj + (I-MINI)*lstri + 1
                  IJKL_INDEX = (L-MINL)*lstrl + (K-MINK)*lstrk  + IJ_INDEX

                  VAL = GHONDO( IJKL_INDEX )
                  IF(ABS(VAL).LT.CUTOFF) CYCLE 
                  I1 = I+LOCI
                  J1 = J+LOCJ
                  K1 = K+LOCK
                  LL1 = L + LOCL

                  VAL1 = VAL*XVAL1
                  VAL4 = VAL*XVAL4

                  NINTN = NINTN + 1
!$omp atomic update
                  FA2d(I1,J1) = FA2d(I1,J1) + VAL4*DA2d(K1,LL1)
!$omp atomic update
                  FA2d(K1,LL1) = FA2d(K1,LL1) + VAL4*DA2d(I1,J1)
!$omp atomic update
                  FA2d(I1,K1) = FA2d(I1,K1) - VAL1*DA2d(J1,LL1)
!$omp atomic update
                  FA2d(J1,LL1) = FA2d(J1,LL1) - VAL1*DA2d(I1,K1)
!$omp atomic update
                  FA2d(I1,LL1) = FA2d(I1,LL1) - VAL1*DA2d(J1,K1)
!$omp atomic update
                  FA2d(J1,K1) = FA2d(J1,K1) - VAL1*DA2d(I1,LL1)
            ENDDO
         ENDDO
      ENDDO
    ENDDO
 return
 end
 end module mod_dirfck



      PROGRAM J6_FOCK
      use OMP_LIB
      use mod_dirfck

      implicit none

      integer :: ishell,jshell,kshell,lshell,nshell,minshell
      integer :: MINI,MINJ,MINK,MINL,MAXI,MAXJ,MAXK,MAXL, &
                 LOCI,LOCJ,LOCK,LOCL,NINTN
      integer :: I,J,K,L,I1,J1,K1,LL1,I_INDEX,IJ_INDEX,IJK_INDEX,IJKL_INDEX
      integer :: lstri,lstrj,lstrk,lstrl,len1,len2,len3,len4
      integer :: L1,L2,maxg,Nerr
      integer :: maxthreads,nteams,limitthread
      double precision :: xval1,xval4,val,val1,val4,diff 
      integer,allocatable :: KTYPE(:),KMIN(:),KMAX(:),KLOC(:)
      double precision,allocatable :: FA2d(:,:),Da2d(:,:),GHONDO(:)

      double precision,allocatable :: FA(:),DA(:),fa_ref(:)
     
      integer,parameter :: MXSH=5000 
      double precision,parameter :: F04=4.0D+00,cutoff = 1.00E-010,&
                                    HFSCAL=1.0D+00,CSCALT=1.0D+00,&
                                  one_eight=0.125D+00,threshold=1.0D-10,&
                                    zero=0.0D+00
      character :: junk1,junk2

      double precision :: t0,t1,t_total,t0_fock,t1_fock,t_fock

!      maxthreads = omp_get_max_threads()
!      maxthreads = 1
!      call omp_set_num_threads(maxthreads) 

      nteams = 1
      limitthread = 1

!      write(6,*) 'total num of thread=',maxthreads
      write(6,*) '         num_teams =',nteams
      write(6,*) '      thread_limit =',limitthread

      t0 = omp_get_wtime()

! dimension for basis set
      read(5,*) junk1, junk2, L1
      read(5,*) junk1, junk2, nshell
      L2= (L1*L1+L1)/2
      allocate(FA(L2))
      allocate(FA2D(L1,L1))
      allocate(DA2D(L1,L1))
      allocate(KTYPE(nshell))
      allocate(KLOC(nshell))
      allocate(KMIN(nshell))
      allocate(KMAX(nshell))
      allocate(DA(L2))
      allocate(FA_ref(L2))

      FA = zero; FA2D = zero; DA2D = zero
      KTYPE = 0; KLOC = 0; KMIN = 0; KMAX = 0 
      nintn = 0
! read KTYPE,KLOC,KMIN,KMAX
      read(5,*) junk1, junk2, nshell
      call RDMATINT(KTYPE,nshell)
 
      call BASCHK(maxg,len1,len2,len3,len4,nshell,ktype)

      read(5,*) junk1, junk2, nshell
      call RDMATINT(KLOC,nshell)
  
      read(5,*) junk1, junk2, nshell
      call RDMATINT(KMIN,nshell)

      read(5,*) junk1, junk2, nshell
      call RDMATINT(KMAX,nshell)
       
! read density
      read(5,*) junk1
      call RDMATFLOAT(DA,L2)
      call CPYTSQ(DA,DA2d,L1,1,L2)     

! read ref FA
      read(5,*) junk1
      call RDMATFLOAT(FA_ref,L2)

! set GHONDO to be 1.0
      allocate(GHONDO(maxg))
      GHONDO = 1.0D+00 

      t0_fock = omp_get_wtime()

!$omp target data &
!$omp map(to:kloc,kmin,kmax,DA2d,GHONDO) &
!$omp map(tofrom:FA2d,nintn) 
!!$omp map(to:nshell,len1,len2,len3,len4)

!$omp target teams distribute parallel do collapse(4) &
!!$omp first private(L1,len1,len2,len3,len4) &
!$omp shared(DA2d,FA2d, &
!$omp        kloc,kmin,kmax,nshell,ghondo) &
!$omp private(ishell,jshell,kshell,lshell,lstri,lstrj,lstrk,lstrl,&
!$omp         mini,minj,mink,minl,maxi,maxj,maxk,maxl,&
!$omp         loci,locj,lock,locl,I,J,K,L,I1,J1,K1,LL1, &
!$omp         xval1,xval4,val,val1,val4, &
!$omp         ij_index,ijkl_index) reduction(+:nintn)
      DO ishell = 1,nshell
      DO jshell = 1,nshell
      DO kshell = 1,nshell
      DO lshell = 1,nshell


      lstri = len4
      lstrj = len3
      lstrk = len2
      lstrl = len1
      call dirfck(FA2d,Da2d,GHONDO, KTYPE,KMIN,KMAX,KLOC, &
                 ishell,jshell,kshell,lshell,nshell,minshell, &
                 lstri,lstrj,lstrk,lstrl,L1,maxg)


!      MINI = KMIN(ishell)
!      MINJ = KMIN(jshell)
!      MINK = KMIN(kshell)
!      MINL = KMIN(lshell)
!      MAXI = KMAX(ishell)
!      MAXJ = KMAX(jshell)
!      MAXK = KMAX(kshell)
!      MAXL = KMAX(lshell)
!      LOCI = KLOC(ishell)-MINI
!      LOCJ = KLOC(jshell)-MINJ
!      LOCK = KLOC(kshell)-MINK
!      LOCL = KLOC(lshell)-MINL
!      XVAL1 = HFSCAL
!      XVAL4 = F04*CSCALT
!
!      DO I = MINI,MAXI
!         DO J = MINJ,MAXJ
!            DO K =  MINK,MAXK
!               DO L=MINL,MAXL
!
!                  IJ_INDEX = (J-MINJ)*lstrj + (I-MINI)*lstri + 1
!                  IJKL_INDEX = (L-MINL)*lstrl + (K-MINK)*lstrk  + IJ_INDEX
!
!                  VAL = GHONDO( IJKL_INDEX )
!                  IF(ABS(VAL).LT.CUTOFF) CYCLE 
!                  I1 = I+LOCI
!                  J1 = J+LOCJ
!                  K1 = K+LOCK
!                  LL1 = L + LOCL
!
!                  VAL1 = VAL*XVAL1
!                  VAL4 = VAL*XVAL4
!
!                  NINTN = NINTN + 1
!!$omp atomic update
!                  FA2d(I1,J1) = FA2d(I1,J1) + VAL4*DA2d(K1,LL1)
!!$omp atomic update
!                  FA2d(K1,LL1) = FA2d(K1,LL1) + VAL4*DA2d(I1,J1)
!!$omp atomic update
!                  FA2d(I1,K1) = FA2d(I1,K1) - VAL1*DA2d(J1,LL1)
!!$omp atomic update
!                  FA2d(J1,LL1) = FA2d(J1,LL1) - VAL1*DA2d(I1,K1)
!!$omp atomic update
!                  FA2d(I1,LL1) = FA2d(I1,LL1) - VAL1*DA2d(J1,K1)
!!$omp atomic update
!                  FA2d(J1,K1) = FA2d(J1,K1) - VAL1*DA2d(I1,LL1)
!            ENDDO
!         ENDDO
!      ENDDO
!    ENDDO
!
       ENDDO
       ENDDO
       ENDDO
       ENDDO
       !$omp end target teams distribute parallel do 

       !$omp target update from(fa2d)
!$omp end target data

      t1_fock = omp_get_wtime()
      t_fock  = t1_fock - t0_fock
   
      CALL DSCAL(L1*L1,one_eight,FA2d,1)

      CALL CPYSQT(fa2d,fa,L1,1,L2)
      Nerr = 0
      DO I = 1, L2
         diff = abs(fa(I) - fa_ref(I))
         IF(diff.GE.threshold) THEN
              !WRITE(6,FMT=100) I,FA_ref(I),diff
              Nerr = Nerr + 1
         ENDIF
      ENDDO
      if(nerr.ne.0) write(6,*) 'INCORRECT!!'
      IF(Nerr.EQ.0) write(6,*) 'FA ALL CORRECT!'

 100  FORMAT('FA(',I4,')= ',F12.9,'  INCORRECT with error ',E11.4)
       
      t1 = omp_get_wtime()
      t_total = t1 - t0

      write(6,*) 'fock build time (s):',t_fock
      write(6,*) 'total time(s):',t_total

      deallocate(ghondo,fa,fa2d,da,da2d,fa_ref,ktype,kloc,kmin,kmax)

      STOP
      END

