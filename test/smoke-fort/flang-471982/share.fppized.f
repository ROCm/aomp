      module share

      BYTE, dimension (:), allocatable::rho_i
      
!$omp declare target(rho_i)

      public :: allocfield
      contains
      subroutine allocfield

      ! ok    1966079 - 16#1DFFFF 2#111011111111111111111
      ! crash 1966080 - 16#1E0000 2#111100000000000000000
      
      allocate (rho_i(0:1966080))

      end subroutine allocfield

      end module share

