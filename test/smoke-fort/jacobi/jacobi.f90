module jacobi_mod
  use, intrinsic :: ISO_Fortran_env, only : int32, real64, int64, stdout=>output_unit
  use input_mod, only: debug
  use mesh_mod, only: mesh_t
  use norm_mod, only: norm
  use laplacian_mod, only: laplacian
  use boundary_mod, only: boundary_conditions
  use update_mod, only: update
  use omp_lib, only: omp_get_wtime
  implicit none

  private
  public :: jacobi_t, init_jacobi, run_jacobi

  real(real64), parameter :: pi = 4._real64*atan(1._real64)
  real(real64), parameter :: tolerance = 1.e-5_real64
  integer(int32) :: max_iters = 1

  type :: jacobi_t
    !type(mesh_t), pointer :: p_mesh
    real(real64), allocatable :: u(:,:), rhs(:,:), au(:,:), res(:,:)
    real(real64) :: t_start, t_stop, elapsed
    integer(int32) :: iters
  end type jacobi_t

contains

  subroutine init_jacobi(this,mesh)
    type(jacobi_t), intent(inout) :: this
    type(mesh_t), intent(inout) :: mesh
    integer(int32) :: i,j
    real(real64) :: rhs_bc

    !mesh=>mesh
    allocate(this%u(mesh%n_x,mesh%n_y))
    allocate(this%au(mesh%n_x,mesh%n_y))
    allocate(this%rhs(mesh%n_x,mesh%n_y))
    allocate(this%res(mesh%n_x,mesh%n_y))

    do j=1,mesh%n_y
      do i=1,mesh%n_x
        this%u(i,j)=0._real64 ! initial guess
        this%rhs(i,j)=0._real64 ! forcing function
      end do
    end do

    do i=1,mesh%n_x
      rhs_bc = cos(pi*mesh%x(i))/mesh%dx**2
      this%rhs(i,1) = this%rhs(i,1) + rhs_bc
      this%rhs(i,mesh%n_y) = this%rhs(i,mesh%n_y) + rhs_bc
    end do

    do j=1,mesh%n_y
      rhs_bc = cos(pi*mesh%y(j))/mesh%dy**2
      this%rhs(1,j) = this%rhs(1,j) + rhs_bc
      this%rhs(mesh%n_x,j) = this%rhs(mesh%n_x,j) + rhs_bc
    end do

    if (debug) then
      max_iters=10
      write(stdout,'(*(E12.3E2))') mesh%x
      write(stdout,'(*(E12.3E2))') mesh%y
      write(*,*)
      call print_2D(this%rhs)
      write(*,*)
    end if

    this%au = 0._real64
    this%res = this%rhs
    !write(stdout,*) this%res
    !!$omp target enter data map(to:mesh%n_x,mesh%n_y,mesh%dx,mesh%dy)
    !$omp target enter data map(to:this%u,this%rhs,this%au,this%res)
    !!$omp target enter data map(to:this,this%u,this%rhs,this%au,this%res)
    !!$omp target enter data map(to:this)

    !!$omp target update from(this%res)
    !write(stdout,*) this%res

  end subroutine init_jacobi

  subroutine run_jacobi(this,mesh)
    type(jacobi_t), intent(inout) :: this
    type(mesh_t), intent(inout) :: mesh

    real(real64) :: resid

    write(stdout,'(A)') 'Starting Jacobi run'
    this%iters = 0

    resid = norm(mesh, this%res)
    write(stdout,'(A,I4,A,ES11.5)') 'Iteration: ',this%iters,' - Residual: ',resid
    
    this%t_start = omp_get_wtime()

    do while (this%iters < max_iters .and. resid > tolerance)
      call laplacian(mesh,this%u,this%au)
      !!$omp target update from(this%au)
      !write(stdout,*) this%au
      if (debug) then
        call print_2D(this%au)
        write(stdout,*)
      end if
      call boundary_conditions(mesh,this%u,this%au)
      !!$omp target update from(this%au)
      !write(stdout,*) this%au
      if (debug) then
        call print_2D(this%au)
        write(stdout,*)
      end if
      call update(mesh,this%rhs,this%au,this%u,this%res)
      !!$omp target update from(this%u)
      !write(stdout,*) this%u
      if (debug) then
        call print_2D(this%u)
        write(stdout,*)
      end if
      !!$omp target update from(this%res)
      !write(stdout,*) this%res
      if (debug) then
        call print_2D(this%res)
        write(stdout,*)
        write(stdout,*)
      end if
      resid = norm(mesh,this%res)
      this%iters = this%iters + 1
      if (debug) write(stdout,'(A,I4,A,ES11.5)') 'Iteration: ',this%iters,' - Residual: ',resid
      if (mod(this%iters,100) == 0 .and. .not. debug) write(stdout,'(A,I4,A,ES11.5)') 'Iteration: ',this%iters,' - Residual: ',resid
    end do

    this%t_stop = omp_get_wtime()
    this%elapsed = this%t_stop - this%t_start

    write(stdout,'(A,I4,A,ES11.5)') 'Stopped after ',this%iters,' iterations with residue: ',resid

    call print_results(this,mesh)

  end subroutine run_jacobi

  subroutine print_2D(array)
    real(real64), intent(in) :: array(:,:)

    integer(int32) :: row, array_shape(2)

    array_shape = shape(array)
    do row = 1, array_shape(1)
      write(stdout,'(*(E12.3E2))') array(row,:)
    end do
  end subroutine print_2D

  subroutine print_results(this,mesh)
    type(jacobi_t), intent(in) :: this
    type(mesh_t), intent(in) :: mesh
    real(real64) :: lattice_updates, flops, bandwidth

    write(stdout,'(A,ES8.2,A)') 'Total Jacobi run time: ',this%elapsed,' sec.'

    lattice_updates = real(mesh%n_x,real64)*mesh%n_y*this%iters
    flops = 17._real64*lattice_updates
    bandwidth = 12._real64*lattice_updates*real64

    write(stdout,'(A,EN11.3,A)') 'Measured lattice updates: ',lattice_updates/this%elapsed,' LU/s'
    write(stdout,'(A,EN11.3,A)') 'Measured FLOPS: ',flops/this%elapsed,' FLOPS'
    write(stdout,'(A,EN11.3,A)') 'Measured device bandwidth: ',bandwidth/this%elapsed,' B/s'
    write(stdout,'(A,F5.3)') 'Measured AI=',flops/bandwidth

  end subroutine print_results

end module jacobi_mod
