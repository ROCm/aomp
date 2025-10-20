program SAXPY_COMPLEX
INTEGER,PARAMETER :: n=10
integer :: i
complex :: a
complex,  dimension(n) :: x
complex, dimension(n) :: y

a=(0, 2)
print*,a
do i =1,n
    x(i) = CMPLX(i,i)
end do
y=0.0d0


call saxpy(a, x, y, n)

do i =1,n
    print*, x(i),y(i)
end do


end program


subroutine saxpy(a, x, y, n)
IMPLICIT NONE
integer :: n, i
complex :: a
complex, dimension(n) :: x
complex, dimension(n) :: y
!$omp target
do i=1,n
y(i) = x(i)/a + y(i)
end do
!$omp end target
end subroutine
