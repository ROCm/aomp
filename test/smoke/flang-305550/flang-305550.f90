MODULE wave_struct_def
TYPE wavedes
        INTEGER,POINTER :: LMMAXX(:)
        END TYPE wavedes

TYPE wavedes1
        INTEGER,POINTER :: LMMAXX(:) => NULL()
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
      ALLOCATE(OUTPUT(N))

      do i=1, N
      WDES%LMMAXX(i) = 1
      OUTPUT(i) = 0
      enddo
      !$OMP TARGET ENTER DATA MAP(TO:WDES)
      !$OMP TARGET ENTER DATA MAP(TO:WDES%LMMAXX)

      ! use WDES / WDES%LMMAXX in different loops/directives

      WDES1%LMMAXX => WDES%LMMAXX

      !$OMP TARGET ENTER DATA MAP(TO:WDES1)
      !$OMP TARGET ENTER DATA MAP(TO:WDES1%LMMAXX)
      
      !$OMP TARGET TEAMS DISTRIBUTE MAP(FROM:OUTPUT)
      do i=1, N
          OUTPUT(i) = WDES%LMMAXX(i)
      enddo
      !$OMP END TARGET TEAMS DISTRIBUTE

      do i=1, N
      if (OUTPUT(i) .ne. 1) then
        write(*,*)"ERROR: wrong answer"
        stop 2
      endif
      write(*,*) "OUTPUT(", i, ")=", OUTPUT(i)
      enddo
      !$OMP TARGET EXIT DATA MAP(DELETE:WDES1%LMMAXX)
      !$OMP TARGET EXIT DATA MAP(DELETE:WDES1)

      !$OMP TARGET EXIT DATA MAP(DELETE:WDES%LMMAXX)
      !$OMP TARGET EXIT DATA MAP(DELETE:WDES)
      write(*,*) "SUCCESS"
      return

 end program test

