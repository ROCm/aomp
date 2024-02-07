module test_0
        implicit none
        integer :: decltar_v = 5
    !$omp declare target link(decltar_v)
end module test_0
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
        use test_0
        integer :: int_array(*)
        integer :: array_length  
        integer :: new_len
        integer :: v = 5 ! global, when given value like this
        ! v = 5 ! local, when given value like this 
!$omp target map(tofrom:new_len) map(tofrom:decltar_v)
        new_len = decltar_v + v 
!$omp end target
        print*, new_len
        do index_ = 1, new_len 
                int_array(index_) = index_  
        end do
end subroutine writeIndex
