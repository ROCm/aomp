PROGRAM test_sections_target_with_allocatable
implicit none
integer,parameter :: N=100, rt=8
real(kind=rt), allocatable, dimension(:) :: a

allocate(a(1:N))

!$omp parallel
!$omp target
    a(1) = 1.0_rt
!$omp end target
!$omp end parallel

END PROGRAM
