program link
implicit none
real :: r
real, dimension(5) :: xs

!$omp target map(xs, r)
xs = 2
xs = modulo(xs, 3)
r = cosh(r)
r = tanh(r)
!$omp end target

end program
