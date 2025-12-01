!
!Copyright © Advanced Micro Devices, Inc., or its affiliates.
!
!SPDX-License-Identifier:  MIT
! 
program repro
  implicit none
  integer                         :: k
  real, allocatable, dimension(:) :: data1

  allocate(data1(1))
  data1(1) = 1

  !$omp target teams distribute parallel do map(tofrom: data1)
  do k=1,1
    call matrix_vector_code(42., data1)
  end do
  !$omp end target teams distribute parallel do 

  print *, ">>>> data1=", data1

contains

  subroutine matrix_vector_code(cell, lhs)
    implicit none
    real,               intent(in)    :: cell
    real, dimension(1), intent(inout) :: lhs
    lhs(1) = cell
  end subroutine matrix_vector_code
end program repro
