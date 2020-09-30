subroutine MATH(a,b)
        real*8 :: a,b
        print *, cos(b) + sin(b) + sqrt(b) + exp(b) + a**b
   !$omp target map(tofrom:a)
        a = cos(b) + sin(b) + sqrt(b) + exp(b) + a**b
   !$omp end target 
        end subroutine
program main
   REAL*8 :: x,y
   x = 2.0
   y = 12.0
   call MATH(x,y)
   print *,x
end program
