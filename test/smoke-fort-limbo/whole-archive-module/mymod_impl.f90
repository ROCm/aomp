submodule (mymod) mymod_impl
  implicit none
 
contains

  module subroutine vadd(a, b, c, N)
        implicit none
        real :: a(N), b(N), c(N)
        integer :: N, i

    !$omp target map(to: a,b) map(from: c)
    !$omp teams distribute parallel do 
        do i=1,N
            c(i) = a(i) + b(i)
        end do
    !$omp end target
    end subroutine
      
end submodule mymod_impl
