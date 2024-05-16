program main
        integer :: hostArray(100,100), errors = 0
        do i = 1, 100
          do j = 1, 100
                hostArray(j, i) = 0
          end do
        end do
        call writeIndex(hostArray)
        do i = 1, 100
          do j = 1, 100
                if ( hostArray( j, i) /= (i + j) ) then
                        errors = errors + 1
                end if
          end do
        end do
        if ( errors /= 0 ) then
                stop 1
        end if
        print*, "======= FORTRAN Test passed! ======="
end program main

subroutine writeIndex(int_array)
        integer :: int_array(100,100)
!$omp target teams distribute parallel do map(from:int_array) collapse(2)
        do i = 1, 100
          do j = 1, 100
             int_array( j, i) = i + j
          end do
        end do
!$omp end target teams distribute parallel do

end subroutine writeIndex
