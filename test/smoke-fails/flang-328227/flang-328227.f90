MODULE wave_struct_def
TYPE wavedes
        INTEGER,POINTER :: LMMAXX(:)
        REAL,POINTER,CONTIGUOUS :: DATAKE(:,:,:)
        END TYPE wavedes

TYPE wavedes1
        INTEGER,POINTER :: LMMAXX(:) => NULL()
        REAL,POINTER,CONTIGUOUS :: DATAKE(:,:)=>NULL()
        END TYPE wavedes1
END MODULE wave_struct_def


program test
      use wave_struct_def
      integer :: i, j, k, N
      TYPE (wavedes)  WDES
      TYPE (wavedes1)  WDES1
      INTEGER,POINTER :: OUTPUT(:)
      N=10

      ALLOCATE(WDES%LMMAXX(N))
      ALLOCATE(WDES%DATAKE(N,N,N))
      ALLOCATE(OUTPUT(N))

      do i=1, N
      WDES%LMMAXX(i) = 1
      OUTPUT(i) = 0
      enddo

      do i=1, N
      do j=1, N
      do k=1, N
      WDES%DATAKE(i,j,k) = 1
      enddo
      enddo
      enddo

      !$OMP TARGET ENTER DATA MAP(TO:WDES)
      !$OMP TARGET ENTER DATA MAP(TO:WDES%LMMAXX)
      !$OMP TARGET ENTER DATA MAP(TO:WDES%DATAKE)

!$omp target
      WDES1%LMMAXX => WDES%LMMAXX
      WDES1%DATAKE => WDES%DATAKE(:,:,1)
!$omp end target


      !$OMP TARGET PARALLEL DO MAP(FROM:OUTPUT)
      do i=1, N
      OUTPUT(i) = WDES1%DATAKE(i) !WDES1%LMMAXX(i)
      enddo
      !$OMP END TARGET PARALLEL DO

      do i=1, N
      write(*,*) "OUTPUT(", i, ")=", OUTPUT(i)
      enddo

      !$OMP TARGET EXIT DATA MAP(DELETE:WDES%LMMAXX)
      !$OMP TARGET EXIT DATA MAP(DELETE:WDES%DATAKE)
      !$OMP TARGET EXIT DATA MAP(DELETE:WDES)

 end program test
