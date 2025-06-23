program test
  implicit none
  integer,parameter :: np = 100
  integer :: I,J
  real,allocatable,dimension(:,:) :: A

  allocate(A(np,np))
  A=0

  !$omp target enter data map(alloc:A)
  !$omp target teams distribute map(present,to:A)
  do i=1,np
    
    !$omp parallel do
    do j=1,np
      A(j,i) = j + (i - 1) * np
    end do
    
    if(i < 5) CYCLE
    
    !$omp parallel do
    do j=1,np
      A(j,i) = A(j,i) + abs(A(i,j))
    end do
  
  end do
  !$omp target exit data map(release:A)
  deallocate(A)

end program test
