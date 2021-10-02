module test
      contains
        subroutine fib(R)
        REAL(8),DIMENSION(2) :: R
        R(1)=10
        R(2)=12

        end subroutine

        subroutine fib_caller
        complex(8) :: N
        REAL(8),DIMENSION(2) :: R
        equivalence(N, R)
        N=(8,6)
        call fib(R)
        write(*, *) "N=", N
        end subroutine fib_caller
      end module test

program myprogram
        use test
        complex(8) :: M
        call fib_caller
        end program myprogram

