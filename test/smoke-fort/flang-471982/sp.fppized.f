!---------------------------------------------------------------------
      program SP
!---------------------------------------------------------------------
! .../llvm/bin/flang-new -O3 -fopenmp --offload-arch=gfx90a share.fppized.f sp.fppized.f -o spF
!---------------------------------------------------------------------
      use share
      BYTE i
      
      call allocfield

!$omp target parallel do
      do i = 0, 0
         rho_i(0) = 9
      end do

      end
