module norm_mod
  use, intrinsic :: ISO_Fortran_env, only: int32, real64
  use mesh_mod
  implicit none

  private
  public :: norm

contains

  function norm(mesh, u) result(norm_val)
    type(mesh_t), intent(inout) :: mesh
    real(real64), intent(inout) :: u(:,:)
    real(real64) :: norm_val
    integer(int32) :: i,j,n_x,n_y
    real(real64) :: dx,dy

    n_x = mesh%n_x
    n_y = mesh%n_y
    dx = mesh%dx
    dy = mesh%dy

    norm_val = 0._real64

    !!$omp target update from(u)
    !$omp target teams distribute parallel do collapse(2) reduction(+:norm_val)
    do j = 1,n_y
      do i = 1,n_x
        norm_val = norm_val + u(i,j)**2*dx*dy
      end do
    end do

    norm_val = sqrt(norm_val)/(mesh%n_x*mesh%n_y)
  end function norm

end module norm_mod
