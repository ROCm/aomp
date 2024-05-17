program main
        integer :: hostArray(10), errors = 0
        do i = 1, 10
                hostArray(i) = 0
        end do
        call writeIndex(hostArray)
        do i = 1, 10
                if ( hostArray(i) /= i ) then
                        errors = errors + 1
                end if
        end do
        if ( errors /= 0 ) then
                stop 1
        end if
        print*, "======= FORTRAN Test passed! ======="
end program main
subroutine writeIndex(int_array)
        integer :: int_array(*)
!$omp target teams distribute map(tofrom:int_array(1:10))
        do index_ = 1, 10
          int_array(index_) = index_
        end do
!$omp end target teams distribute

end subroutine writeIndex

