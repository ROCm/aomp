program test_alias
  implicit none
  integer:: a(100)
  integer:: b(100)
  integer:: c(100)
  integer:: ii  
  !$omp target 
  do ii=1,100
      c(ii) = b(ii) + a(ii)
  end do
  !$omp end target 
end program test_alias

