subroutine MATH(a,b)
        real*8 :: a,b
   !$omp target map(tofrom:a)
        a = cos(b)
   !$omp end target 
        end subroutine
program main
   REAL*8 :: x
   call MATH(x,12.0)
   print *,x
end program
