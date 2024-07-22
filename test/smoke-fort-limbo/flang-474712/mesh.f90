module mesh_mod
  use, intrinsic :: ISO_Fortran_env, only: int32, real64, stdout=>output_unit
  implicit none

  private
  public :: mesh_t, init_mesh

  real(real64), parameter :: x_min=-0.5_real64, x_max=0.5_real64
  real(real64), parameter :: y_min=-0.5_real64, y_max=0.5_real64
  real(real64), parameter :: l_x=x_max-x_min, l_y=y_max-y_min

  type :: mesh_t
    integer(int32) :: n_x, n_y
    real(real64) ::dx, dy
    real(real64), allocatable :: x(:), y(:)
  end type mesh_t

contains

  subroutine init_mesh(this,n_x,n_y)
    class(mesh_t), intent(inout) :: this
    integer(int32) :: n_x, n_y, i

    this%N_x = n_x
    this%N_y = n_y
    this%dx = l_x/(n_x+1._real64)
    this%dy = l_y/(n_y+1._real64)

    allocate(this%x(0:n_x+1),this%y(0:n_y+1))
    this%x(0) = x_min
    do i = 1,n_x
      this%x(i) = x_min + i*this%dx
    end do
    this%x(n_x+1) = x_max
    this%y(0) = y_min
    do i = 1,n_y
      this%y(i) = y_min + i*this%dy
    end do
    this%y(n_y+1) = y_max

    write(stdout,'(A,I5,A,I5)') 'Domain size: ',n_x,' x ',n_y

  end subroutine init_mesh

end module mesh_mod
