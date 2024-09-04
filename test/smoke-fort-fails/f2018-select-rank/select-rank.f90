program my_select
    implicit none
    integer :: a0 = 0
    integer :: a1(2) = 1
    integer :: a2(2,2) = 2
    integer :: a3(2,2,2) = 3
    integer :: a4(2,2,2,2) = 4
    integer :: a5(2,2,2,2,2) = 5
    integer :: a6(2,2,2,2,2,2) = 6
    integer :: a7(2,2,2,2,2,2,2) = 7

    print *, a0
    call chk_rank(a0)
    print *, a1(1)
    call chk_rank(a1)
    print *, a2(1,1)
    call chk_rank(a2)
    print *, a3(1,1,1)
    call chk_rank(a3)
    print *, a4(1,1,1,1)
    call chk_rank(a4)
    print *, a5(1,1,1,1,1)
    call chk_rank(a5)
    print *, a6(1,1,1,1,1,1)
    call chk_rank(a6)
    print *, a7(1,1,1,1,1,1,1)
    call chk_rank(a7)

contains
    subroutine chk_rank(val)
        implicit none
        integer val(..)
        print *, "chk_rank"
        print *, "rank = ", rank(val)
        select rank(val)
            rank (0)
                print *, "select rank 0"
            rank (1)
                print *, "select rank 1"
            rank (2)
                print *, "select rank 2"
            rank (3)
                print *, "select rank 3"
            rank (4)
                print *, "select rank 4"
            rank default
                print *, "select default", rank(val)
       end select
    end subroutine
end program
