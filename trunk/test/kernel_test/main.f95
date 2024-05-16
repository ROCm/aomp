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
        integer :: new_len
!$omp target map(from:new_len)
        new_len = 10
!$omp end target
        do index_ = 1, new_len 
                int_array(index_) = index_  
        end do
end subroutine writeIndex
