program main
   integer :: a(256)
   integer :: b(256)
   integer :: c
   integer :: d
   integer :: e(256)
   integer :: f
   integer :: i
   !$omp target teams distribute parallel do map(tofrom: a)
   DO i = 1, 256
         a(i) = i
   END DO 
   !$omp end target teams distribute parallel do

   !$omp target teams distribute map(tofrom: b)
   DO i = 1, 256
         b(i) = i
   END DO 
   !$omp end target teams distribute

   !$omp target parallel map(tofrom: c)
         c = 2
   !$omp end target parallel

   !$omp target teams map(tofrom: d)
         d = 3
   !$omp end target teams

   !$omp target parallel do map(tofrom: e)
    DO i = 1, 256
         e(i) = i
    END DO
   !$omp end target parallel do 

   !$omp target map(tofrom : f)
        f = 4
   !$omp end target
   if (f .ne. 4 .or. e(256) .ne. 256) then
     print *, "Failed"
     stop 2
   endif
   print *,"passed"
end program main
