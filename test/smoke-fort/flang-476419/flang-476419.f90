module test
  DOUBLE PRECISION, ALLOCATABLE, DIMENSION(:) :: UNEW
  ! doesn't work with declare target(), which is effectively a to/enter clause. 
  ! This still needs supported however, which Raghu is working on.
  ! It does work without the declare target to however, if we comment it out.
  !$omp declare target(UNEW)

  ! This reproducer, will also work if we remove the "declare target to" and 
  ! replace it with a "declare target link" which is currently implemented to
  ! a more reasonable extent so far. However, in the larger example, we 
  ! encounter another bug/issue that seems unrelated, but likely will crop 
  ! up elsewhere and need addressed. I have a feeling it might be due to 
  ! never having tested declare target link with allocatables/descriptor types
  ! which tend to be a bit of a different beast from the simpler types.
  !!$omp declare target link(UNEW)
end module test

PROGRAM main
use test 

ALLOCATE(UNEW(10))

! AG selfnote: does declare target need any magic inside of a host mapping region like it 
! does in a target region, perhaps clarify with a clang example..?
!$omp target data map(alloc:UNEW)

! changing the 2 to a 1 will circumvent the error, it's the 
! second iteration that causes the issue (as it's a do it 
! always will loop once) as it appears we lose/deallocate the 
! data with the first update from map.
DO NCYCLE = 1, 2
print *, "executing update"
!$omp target update from(UNEW)
end do

!$omp end target data
END PROGRAM

