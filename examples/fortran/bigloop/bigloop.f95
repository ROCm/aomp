
program main
  PARAMETER (nsize=10000)
  real a(nsize, nsize), b(nsize, nsize), c(nsize, nsize)
  integer *8 i,j
  integer *4, dimension(8) :: values, values1

  do i=1,nsize
    do j=1,nsize
      a(i,j) = 0
      b(i,j) = i+j
      c(i,j) = 10
    end do
  end do

  call date_and_time(VALUES=values)
  call foo(a,b,c)
  call date_and_time(VALUES=values1)
  print *,"Time Sec=" , ((values1(6)-values(6))*60*1000 + (values1(7)-values(7)) * 1000+(values1(8)-values(8))) *.001
  write(6,*)"a(1,1)=", a(1,1), "    a(2,2)=", a(2,2)
  if (a(1,1).ne.21 .or. a(2,2).ne.42) then
    stop "ERROR: wrong answers"
  endif
  return
end
subroutine foo(a,b,c)
  PARAMETER (nsize=10000)
  real a(nsize,nsize), b(nsize,nsize), c(nsize,nsize)
  integer i,j,k,omp_get_num_threads,omp_get_thread_num,omp_get_max_threads
!$omp target teams distribute parallel do simd map(from:a) map(to:b,c) private(i,j,k)
  do k=1,20
   do i=1,nsize-1
    do j=1,nsize-1
      a(i,j) = b(i,j) * c(i,j) + i
    end do
   end do
  end do
!$omp end target teams distribute parallel do simd
  return
end

