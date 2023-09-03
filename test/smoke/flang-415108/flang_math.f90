subroutine MATH(a,b)
        real*4 :: a,b
        print *, atan(b)
   !$omp target map(tofrom:a)
        a = atan(b)
   !$omp end target 
        end subroutine
program main
   REAL*4 :: x,y
   x = 2.0
   y = 12.0
   call MATH(x,y)
   print *,x
end program
