program main  
        integer :: hostArray(10), errors = 0  
        do i = 1, 10    
                hostArray(i) = 0  
        end do  
        call writeIndex(hostArray, 10)
        do i = 1, 10    
                if ( hostArray(i) /= i ) then
                        errors = errors + 1    
                end if  
        end do  
        if ( errors /= 0 ) then
                stop 1  
        end if
        print*, "======= FORTRAN Test passed! ======="
!       stop 0
end program main
subroutine writeIndex(int_array, array_length)
        integer :: int_array(*)
        integer :: array_length
!$omp target teams distribute parallel do map(tofrom:int_array(1:10))
        do index_ = 1, 10
          int_array(index_) = index_
        end do
!$omp end target teams distribute parallel do

end subroutine writeIndex

/// CHECK: Tripcount: 10
