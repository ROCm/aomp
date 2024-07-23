program main
        integer :: A(100), B(100), errors = 0
        do i = 1, 100
           A(i) = 0
           B(i) = 0
        end do
        call writeIndex(A,B)
        do i = 1, 100
           if ( A(i) /= i ) then
               errors = errors + 1
           end if
           if ( B(i) /= i ) then
               errors = errors + 1
           end if
        end do
        if ( errors /= 0 ) then
                stop 1
        end if
        print*, "======= FORTRAN Test passed! ======="
end program main

subroutine writeIndex(A, B)
        integer :: A(100)
        integer :: B(100)
!$omp target teams
!$omp distribute parallel do
        do i = 1, 100
           A(i) = i
        end do
!$omp end distribute parallel do

!$omp distribute parallel do
        do i = 1, 100
           B(i) = i
        end do
!$omp end distribute parallel do

!$omp end target teams

end subroutine writeIndex
