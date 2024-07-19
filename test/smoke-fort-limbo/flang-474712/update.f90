module update_mod
  use, intrinsic :: ISO_Fortran_env, only: int32, real64
  use mesh_mod, only: mesh_t
  implicit none

  private
  public :: update

contains

  subroutine update(mesh,rhs,au,u,res)
    type(mesh_t), intent(inout) :: mesh
    real(real64), intent(inout) :: rhs(:,:), au(:,:)
    real(real64), intent(inout) :: u(:,:)
    real(real64), intent(inout) :: res(:,:)
    integer(int32) :: i,j,n_x,n_y
    real(real64) :: temp,factor

    n_x = mesh%n_x
    n_y = mesh%n_y
    factor = (2._real64/mesh%dx**2+2._real64/mesh%dy**2)**-1

    !$omp target teams distribute parallel do collapse(2) private(temp)
    do j = 1,n_y
      do i = 1,n_x
        temp = rhs(i,j) - au(i,j)
        res(i,j) = temp
        u(i,j) = u(i,j) + temp*factor
      end do
    end do
  end subroutine

end module update_mod
