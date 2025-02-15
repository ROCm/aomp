program main
   integer :: a = 0
   !$omp target teams distribute parallel do reduction(+: a)  map(tofrom: a)
   DO i = 1, 256
         a = a + 1
   END DO
   !$omp end target teams distribute parallel do

   if (a .ne. 256 ) then
     print *, "Failed"
     stop 2
   endif
   print *, "Passed"
   return
end program main
