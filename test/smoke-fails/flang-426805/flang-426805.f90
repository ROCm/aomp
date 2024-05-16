Module test
 
contains
 
  subroutine subr
  
  implicit none
 
  !$omp declare target
 
  type mytype
     integer*8 :: l,u
  end type mytype
 
 
  type(mytype) :: a,b
  a%l = 1
  a%u = 2
  b=a
   print *,a%l, a%u
  
end subroutine subr
 
End Module test
program foobar
         use test
         call subr
         end program foobar
