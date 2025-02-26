! This test checks for a problem with HSA_XNACK=1 and target variables
! that are allocatable arrays. It also checks the interaction between
! openmp requires unified_shared_memory, declare target, target data
! map clauses, and whether or not HSA_XNACK is enabled. 
! There are 16 total combinations. 
module global
#ifdef REQUIRES_USM
  !$omp requires unified_shared_memory
#endif
  real, allocatable, dimension(:) :: a_array
  integer :: i_scalar
#ifdef DECLARE_TGT
  !$omp declare target (a_array, i_scalar)
#endif
end module

subroutine initial
  use global
  use descr
  implicit none

  i_scalar = 123
  !$omp target update to(i_scalar)
  allocate(a_array(100))
  a_array = 12
  call print_descr("a_array", a_array)

#ifdef DATA
  !$omp target data map(tofrom:a_array)
#endif
  !$omp target
  call print_descr("a_array", a_array)
  a_array(1) = i_scalar
  !$omp end target
#ifdef DATA
  !$omp end target data
#endif

#ifdef REQUIRES_USM
# define Requires "Y,"
#else
# define Requires "N,"
#endif
#ifdef DECLARE_TGT
# define DclTgt  "Y,"
#else
# define DclTgt  "N,"
#endif
#ifdef DATA
# define Data "Y,"
#else
# define Data "N,"
#endif

  print *, ",xnack,requires usm,target declare,target data,coherent"
  if (a_array(1) /= i_scalar) then
    print *,'XNACK,',XNACK,",",Requires,DclTgt,Data,"N"
  else
    print *,'XNACK,',XNACK,",",Requires,DclTgt,Data,"Y"
  endif
end subroutine 

program p
  implicit none
  call initial
end program
