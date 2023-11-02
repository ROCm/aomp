module var
    implicit none
    real, dimension(:,:),allocatable :: output
end module var

module test_aomp
    use var
    implicit none

contains
    subroutine init_out(nsize)
        integer, intent(in) :: nsize

        allocate(output(nsize, nsize))
    end subroutine init_out

    subroutine do_calc(its,kts)
        real,parameter :: a = 2.0, b = 2.0
        real :: c = 0
        integer :: i, j, nsize
        integer,  intent(in) :: its, kts

        if( command_argument_count() .gt. 0 )then
            nsize = 10
        else
            nsize = 100
        end if
        do i=1,nsize
            do j=1,nsize
                c = c + a/(1+b)
            end do
        end do
#define OFFLOAD_POW_OP
#ifdef OFFLOAD_POW_OP
        !$omp target teams
#endif
        output(its,kts) = 1/c**(a-b)
#ifdef OFFLOAD_POW_OP
        !$omp end target teams
#endif
        
        print *, "output: ", output
    end subroutine do_calc
end module test_aomp

program test
    use test_aomp

    call init_out(100)
    call do_calc(10,23)
    if (output(10,23) .ne. 1.000000) then
        write(*,*)"ERROR: wrong answer"
        stop 2
    endif
    print *,"Luise: ", output(10,23)
    return
end program test
