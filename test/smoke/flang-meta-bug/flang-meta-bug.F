      subroutine vtot_ion_mix()
      implicit none

      real   evx(20)  
      real   eva(20)  
      real   xx
      real   x(20,20)  
      integer     i,j,n
      i=2
      n=20
      eva=0.0d0
#define BUG 1
#if BUG
!$omp  metadirective 
!$omp& when(user={condition(.true..or..false.)}:
!$omp&      simd )
!$omp&      default(loop private(xx))
#else
!$omp       simd 
#endif
        do j=0,n-1
          xx=abs(x(1,i)-x(1,j))
        enddo
      return
      end subroutine vtot_ion_mix
      program main
        call vtot_ion_mix
      end program main
