!
!Copyright © Advanced Micro Devices, Inc., or its affiliates.
!
!SPDX-License-Identifier:  MIT
! 
! example of simple Fortran AMD GPU offloading
program main
  parameter (nsize=10)
  real a(nsize), b(nsize), c(nsize)
  integer i

  do i=1,nsize
    b(i) = i
    c(i) = 10
  end do

!$omp target map(from:a) map(to:b,c)
!$omp parallel do
  do i=1,nsize
    a(i) = b(i) * c(i) + i
  end do
!$omp end target
  print *,a
  return
end
