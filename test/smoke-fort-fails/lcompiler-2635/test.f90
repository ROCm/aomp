program test
implicit none
integer, parameter :: n = 128
integer :: i
real,allocatable :: dx(:)
real :: dx_cpy(n)

allocate(dx(1:n))
dx = 404

!$omp target teams distribute parallel do &
!$omp private(dx)
do i=1,100
dx = 42.0
dx_cpy = dx
enddo

print *, dx_cpy
deallocate(dx)
end program test
