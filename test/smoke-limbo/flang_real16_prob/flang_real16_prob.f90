subroutine size(a,b)
        real*16 :: a,b
        end subroutine
program main
   REAL*16 :: x(2)
   call SIZE(x(1),x(2))
end program
