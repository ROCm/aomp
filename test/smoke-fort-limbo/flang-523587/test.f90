PROGRAM test_usm
      implicit none
      !$omp requires unified_shared_memory
      integer :: i
END PROGRAM
