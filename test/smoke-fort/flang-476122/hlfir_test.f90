module mymod
  implicit none

contains

  subroutine test_omp(dimsB, dimsE, U)
    implicit none

    integer, intent(in) :: dimsB(2), dimsE(2)
    double precision, intent(inout) :: U(dimsB(1):dimsE(1), dimsB(2):dimsE(2))
    integer :: idim1, idim2

    !$OMP TARGET TEAMS DISTRIBUTE PARALLEL DO SIMD COLLAPSE(2)
    do idim1 = dimsB(1), dimsE(1)
      do idim2 = dimsB(2), dimsE(2)
        U(idim1, idim2) = U(idim1, idim2) + 1.0
      END DO
    END DO

  end subroutine test_omp

end module mymod

program main
  use mymod
  implicit none

  integer, parameter :: dimsB(2) = [-2,0]
  integer, parameter :: dimsE(2) = [ 2,4]
  double precision, parameter :: TOL=1.e-12
  double precision, allocatable, dimension(:,:) :: U

  allocate(U(dimsB(1):dimsE(1),dimsB(2):dimsE(2)))
  U(:,:) = 1.0

  !$OMP TARGET ENTER DATA MAP(to: U)
  call test_omp(dimsB, dimsE, U)
  !$OMP TARGET EXIT DATA MAP(from: U)

  if (abs(sum(U) - 50.0d0) < TOL) then
    print*,"OMP offload test passed."
  else
    print*,"OMP offload test failed."
    stop
  end if

  deallocate(U)
end program main
