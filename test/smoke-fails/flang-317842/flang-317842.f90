real function double_dot(a,b) result(dot)
  implicit none

  real, intent(in) :: a(:,:), b(:,:)

  integer :: i,j

  !$omp target teams distribute parallel do reduction(+:dot)
  do j = 1,size(a,dim=2)
    !$omp parallel do reduction(+:dot)
    do i = 1,size(a,dim=1)
      dot = a(i,j)*b(i,j)
    enddo
  enddo

end function
