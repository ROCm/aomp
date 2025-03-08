program superMWE
  use iso_c_binding, only: C_DOUBLE
  implicit none(external)
  integer, parameter :: WP=C_DOUBLE
  ! set up size
  integer :: NX, NY, NZ, NT
  ! counters
  integer :: nnt, nnx, nny, nnz
  ! sum
  real(kind=WP) :: sumTrP, expsumTrP
  integer :: nP, expnP
  NX = 32
  NY = 32
  NZ = 32
  NT = 8
  nP = 0
  sumTrP = 0.0_WP
  do concurrent(nnx=1:nx, nny=1:ny, nnz=1:nz, nnt=1:nt) reduce(+:sumTrP, nP)
         sumTrP = sumTrP + 1.0_WP
         nP = nP + 1
   end do
   write(*,*) 'Done', sumTrP, nP
   expnP = NX * NY * NZ *NT
   expsumTrP = expNP
   write(*,*) 'Expected', expsumTrP, expnP
   if ((np /= expnP) .or. (sumTrP /= expsumTrP)) then
       if (np /= expnP) then
           write(*,*) 'Error np != expnP', nP, expnP
       endif
       if (sumTrP /= expsumTrP) then
           write(*,*) 'Error sumTrP != expsumTrP', sumTrP, expsumTrP
       endif
       stop 1
   endif
end program superMWE
