MODULE thismodule
implicit none

public :: grid_trafo_dummy

integer, parameter :: rk=8

contains

 function grid_trafo_dummy(F_xi, nx, ny, nz) result(F_x)
      real(rk), dimension(:,:,:), contiguous, intent(in) :: F_xi                ! d(F)/d(xi)
      integer,                            intent(in) :: nx, ny, nz          ! Size of input field
      real(rk), dimension(nx,ny,nz)                      :: F_x                 ! Result: x-derivative in physical space
      integer                                        :: i, j, k             ! Loop indices

    !$omp target teams distribute parallel do simd collapse(3)
    do k = 1, nz
      do j = 1, ny
        do i = 1, nx
          F_x(i,j,k) = F_xi(i,j,k)
         end do
       end do
    end do
    !$omp end target teams distribute parallel do simd
  end function

END MODULE

PROGRAM test_stackarray
  use thismodule

  implicit none

  integer :: i,j,k
  integer, parameter :: nx=256,ny=256,nz=256

  real(rk), dimension(nx,ny,nz)                      :: F_x                 ! Result: x-derivative in physical space
  real(rk), dimension(nx,ny,nz)                      :: F_xi                 ! Result: x-derivative in physical space

  F_x = grid_trafo_dummy(F_xi, nx, ny, nz)
END PROGRAM
