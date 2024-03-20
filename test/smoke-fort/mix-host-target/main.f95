program main
       integer :: hostArray(100), errors = 0
       do i = 1, 100
               hostArray(i) = 0
       end do
       call writeIndex(hostArray, 100)
       do i = 1, 100
               if ( hostArray(i) /= i * 100 ) then
                       errors = errors + 1
               end if
       end do
       if ( errors /= 0 ) then
               stop 1
       end if
       print*, "======= FORTRAN Test passed! ======="

end program main
subroutine writeIndex(int_array, array_length)
       integer :: int_array(100)
       integer :: array_length
!$omp target data map(from:int_array)
!$omp  target teams distribute parallel do
       do index_ = 1, 100
         int_array(index_) = index_
       end do
!$omp end target data

!$omp parallel do
       do index_ = 1, 100
         int_array(index_) = int_array(index_)* 100;
       end do

end subroutine writeIndex

// CHECK: ======= FORTRAN Test passed! =======
