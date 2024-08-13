program main
  implicit none

  integer, parameter :: dimsB(2) = [-2,0]
  integer, parameter :: dimsE(2) = [ 2,4]
  double precision, parameter :: TOL=1.e-12
  double precision, allocatable, dimension(:,:) :: U
  integer :: idim1, idim2

  allocate(U(dimsB(1):dimsE(1),dimsB(2):dimsE(2)))
  U(:,:) = 1.0

  !$OMP TARGET ENTER DATA MAP(to: U, dimsB, dimsE)
  !$OMP TARGET TEAMS DISTRIBUTE PARALLEL DO SIMD COLLAPSE(2)
  do idim1 = dimsB(1), dimsE(1)
    do idim2 = dimsB(2), dimsE(2)
      U(idim1, idim2) = U(idim1, idim2) + 1.0
    END DO
  END DO
  !$OMP TARGET EXIT DATA MAP(from: U) MAP(release: dimsB, dimsE)

  if (abs(sum(U) - 50.0d0) < TOL) then
    print*,"OMP offload test passed."
  else
    print*,"OMP offload test failed."
    stop
  end if

  deallocate(U)
end program main
