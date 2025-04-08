module test
  contains
  function ex(a,b,c)
    !$omp declare target(ex)
    integer :: a,b,c
    ex = a + b + c
  end function ex
end module test
