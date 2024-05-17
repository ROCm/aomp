module test
    type :: Float2
        real :: x
        real :: y
    end type
contains
    type(Float2) function AddFloat2(a, b)
        implicit none
        type(Float2), intent(in) :: a, b
        !$omp declare target
        
        AddFloat2 = Float2(a%x + b%x, a%y + b%y)
    end function
end module

program main
    use test
    type(Float2) f1, f2, f3
    f1%x = 1
    f1%y = 2
    f2%x = 3
    f2%y = 4
    f3 = AddFloat2(f1, f2)
    print *, f1
    print *, f2
    print *, f3
    if ( f3%x /= 4 ) then
        print *, "f3%x incorrect"
        stop 1
    end if
    if ( f3%y /= 6 ) then
        print *, "f3%y incorrect"
        stop 1
    end if
    print*, "======= FORTRAN Test passed! ======="
end
