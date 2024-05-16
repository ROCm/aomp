subroutine MATH(a,b)
        real*8 :: a,b
        print *, cos(b) + sin(b) + sqrt(b) + exp(b) + a**b
   !$omp target map(tofrom:a)
        a = cos(b) + sin(b) + sqrt(b) + exp(b) + a**b
   !$omp end target 
        end subroutine
program main
   REAL*8 :: x,y,hostresult
   x = 2.0
   y = 12.0
   hostresult = cos(y) + sin(y) + sqrt(y) + exp(y) + x**y
   call MATH(x,y)
   print *,x
   if (x .ne. hostresult) then
      print *, "Failed: ", x, "!=",hostresult
      stop 2
   endif
   print *, "Passed"
end program
