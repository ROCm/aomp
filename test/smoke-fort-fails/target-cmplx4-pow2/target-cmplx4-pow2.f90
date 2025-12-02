!
!Copyright © Advanced Micro Devices, Inc., or its affiliates.
!
!SPDX-License-Identifier:  MIT
! 
program test
      implicit none
      integer :: i
      integer,parameter :: N = 10 
      complex(4) :: A(N), B(N)
      real(8) :: rdiff, idiff, remax, iemax
      real(8),parameter :: eps = 1.0e-5

      do i = 1, N
        A(i) = i * (2, 1)
        B(i) = A(i) ** 2
      enddo

!$omp target  parallel do map(tofrom:A)
      do i = 1, N
        A(i) = A(i) ** 2
      enddo
!$omp end target parallel do

      write(*, *), "A=", A
      write(*, *), "B=", B

      do i = 1, N
        rdiff = abs(A(i)%re - B(i)%re)
        idiff = abs(A(i)%im - B(i)%im)
        remax = 2 * max(eps, eps * abs(B(i)%re))
        iemax = 2 * max(eps, eps * abs(B(i)%im))
        if ((rdiff > remax) .or. (idiff > iemax)) then
          write(*,*) 'Error A != B', A(i), B(i)
          write(*,*) '  diff:(', rdiff, ', ', idiff, ')'
          write(*,*) '  emax:(', remax, ', ', iemax, ')'
          stop 1
          stop 1
        endif
      enddo
end program test
