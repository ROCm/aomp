program main
        integer :: hostArray(10000), errors = 0, sum = 0, hostsum = 0
        do i = 1, 10000
                hostArray(i) = 0
        end do
        call writeIndex(hostArray, 10000)
        do i = 1, 10000
                hostsum = hostsum + hostArray(i)
                if ( hostArray(i) /= i ) then
                        errors = errors + 1
                end if
        end do
        hostsum = hostsum + 1000
        call computeSum(hostArray, 10000, sum)
        if (hostsum /= sum) then
           stop 1
        end if
        if ( errors /= 0 ) then
                stop 1
        end if
        print*, "======= FORTRAN Test passed! ======="
!       stop 0
end program main
subroutine writeIndex(int_array, array_length)
        integer :: int_array(10000)
        integer :: array_length
        do index_ = 1, 10000
          int_array(index_) = index_
       end do
end subroutine writeIndex

subroutine computeSum(int_array, array_length, sum)
        integer :: int_array(10000)
        integer :: array_length
        integer :: sum
        sum = 1000;
 !$omp target teams distribute parallel do map(to:int_array) map(tofrom:sum) reduction(+:sum)
        do index_ = 1, 10000
          sum = sum + int_array(index_)
        end do
!$omp end target teams distribute parallel do
end subroutine computeSum
