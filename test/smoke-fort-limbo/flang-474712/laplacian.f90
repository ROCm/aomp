module laplacian_mod
  use, intrinsic :: ISO_Fortran_env, only: int32, real64
  use mesh_mod, only: mesh_t
  implicit none

  private
  public :: laplacian

contains

  subroutine laplacian(mesh,u,au)
    type(mesh_t), intent(inout) :: mesh
    real(real64), intent(inout) :: u(:,:)
    real(real64), intent(inout) :: au(:,:)
    integer(int32) :: i,j,n_x,n_y
    real(real64) :: invdx2,invdy2

    n_x = mesh%n_x
    n_y = mesh%n_y
    invdx2 = mesh%dx**-2
    invdy2 = mesh%dy**-2

    !$omp target teams distribute parallel do collapse(2)
    do j = 2,n_y-1
      do i = 2,n_x-1
        au(i,j) = (-u(i-1,j)+2._real64*u(i,j)-u(i+1,j))*invdx2 &
           + (-u(i,j-1)+2._real64*u(i,j)-u(i,j+1))*invdy2
      end do
    end do

  end subroutine laplacian

end module laplacian_mod
