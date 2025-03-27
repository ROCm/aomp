!********************************example code: ************************

PROGRAM test_sections_target_with_allocatable
implicit none

!unified memory is not essential for this reproducer to fail
!$omp requires unified_shared_memory

integer,parameter :: N=100, rt=8
integer :: i
!pointer or allocatable does not make a difference, both fail
real(kind=rt), allocatable, dimension(:) :: a

!but works with a stack array instead of allocatable/pointer!
!real(kind=rt) :: a(N)

allocate(a(1:N))

!$omp sections
!$omp section
!$omp target teams distribute parallel do
do i=1,N
    a(i) = 1.0_rt
end do
!$omp end target teams distribute parallel do
!$omp end sections

END PROGRAM
!*************************************************
