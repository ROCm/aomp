module hip_interface
  
  interface
    subroutine omp_hip(a,b,c,N) bind(C)
      use iso_c_binding
      type(c_ptr), value  :: a, b, c
      integer(c_int), value      :: N
    end subroutine omp_hip
  end interface

end module hip_interface

program example
  use hip_interface
  use iso_c_binding
  implicit none

  real(8), allocatable, target, dimension(:) :: a, b, c
  integer, parameter :: N = 1024

  allocate(a(N),b(N),c(N))
  a = 1.0
  b = 2.0
  c = 0.0

  !$omp target enter data map(to:a,b,c)

  !$omp target data use_device_addr(a,b,c)
  call omp_hip(c_loc(a),c_loc(b),c_loc(c),N)
  !$omp end target data

  !$omp target update from(c)
  !$omp target exit data map(delete:a,b,c)

  write(*,*) "c(1) = ", c(1) ! answer should be 3.0

  if (c(1) /= 3.0) then
     print *, "Answer should be 3.0"
     stop 1
  endif
end program example
