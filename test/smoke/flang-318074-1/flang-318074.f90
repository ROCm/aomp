program modulo_test
  implicit none
  integer(2) :: a,b
  integer :: res_int16
  integer :: c,d, res_int32
  integer(8) :: e,f, res_int64
  real :: g,h, res_float
  real(8) :: i,j, res_double
  a = 1
  b = -2
  c = 3
  d = -4
  e = 5
  f = -6
  g = 7.7
  h = -8.8
  i = 9.9
  j = -10.10
  !$omp target map(tofrom: a,b,res_int16,c,d,res_int32, e,f,res_int64, g,h,res_float, i,j,res_double)
    res_int16 = modulo(a,b)
    res_int32 = modulo(c,d)
    res_int64 = modulo(e,f)
    res_float = modulo(g,h)
    res_double = modulo (i,j)
  !$omp end target

  if (res_int16 .ne. modulo(a,b)) then
    print *, "Failed modulo int16_t"
    stop 2
  endif

  if (res_int32 .ne. modulo(c,d)) then
    print *, "Failed modulo int32_t"
    stop 2
  endif

  if (res_int64 .ne. modulo(e,f)) then
    print *, "Failed modulo int64_t"
    stop 2
  endif

  if (res_float .ne. modulo(g,h)) then
    print *, "Failed modulo float"
    stop 2
  endif

  if (res_double .ne. modulo(i,j)) then
    print *, "Failed modulo double"
    stop 2
  endif
  print *, "Passed"
end program modulo_test
