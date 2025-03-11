program test
implicit none
  integer :: array(10,10),i,j
  !$omp target teams distribute shared(array) map(tofrom:array)
  do i=1,10
    !$omp parallel do
    do j=1,10
      array(j,i)=i+j
    end do
  end do
end program test
