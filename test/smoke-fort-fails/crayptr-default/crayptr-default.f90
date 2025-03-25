program test_crayptr
  implicit none
  integer, parameter :: NP = 8
  integer var(2,*)
  pointer(ivar,var)
  integer pointee(2,NP)
  integer result(NP), verify(NP)
  integer n, n1, n2, npair
  npair = NP

  do n=1,npair
      pointee(1,n) = n
      pointee(2,n) = 100.0 * n
      verify(n) = pointee(1,n) + pointee(2,n)
  enddo
  ivar = loc(pointee)

! Test fails with semantic check with default(none)
! Test passes if default(none) removed
!$omp parallel do default(none)
!$omp& shared (ivar,npair,result)
!$omp& private (n,n1,n2)
  do n=1,npair
    n1=var(1,n)
    n2=var(2,n)
    result(n) = n1 + n2
  enddo

  print *, result
  do n=1,npair
      if (result(n) /= verify(n)) then
          print *, "Failed verify @ result(", n, ") saw:", result(n), " expected:", verify(n)
          stop 1
      endif
  enddo
  print *, "Success"
end program test_crayptr
