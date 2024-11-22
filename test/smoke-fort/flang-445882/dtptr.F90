program main

  use iso_fortran_env

  implicit none

  integer(kind=int32) :: i32_
  integer, parameter :: ccs_int = kind(i32_)

  type :: topology
    integer(ccs_int), dimension(:, :), allocatable :: nb_indices
  end type topology

  type :: ccs_mesh
    type(topology) :: topo
  end type ccs_mesh

  type :: cell_locator
    type(ccs_mesh), pointer :: mesh
    integer(ccs_int) :: index_p
  end type cell_locator

  type :: face_locator
    type(ccs_mesh), pointer :: mesh
    integer(ccs_int) :: index_p
    integer(ccs_int) :: cell_face_ctr
  end type face_locator

  type :: neighbour_locator
    type(ccs_mesh), pointer :: mesh
    integer(ccs_int) :: index_p
    integer(ccs_int) :: nb_counter
  end type neighbour_locator

  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  type(face_locator) :: loc_f
  logical :: is_boundary
  integer :: M, N, i, j
  real :: start, finish

  M = 2000
  N = 10000

  allocate (loc_f%mesh)
  allocate (loc_f%mesh%topo%nb_indices(M,N))

  do i = 1,M
    do j = 1,N
      loc_f%mesh%topo%nb_indices(i,j) = j+2
    end do
  end do

  call cpu_time(start)
  do i = 1,M
    do j = 1,N
      loc_f%cell_face_ctr = i
      loc_f%index_p = j
      call get_boundary_status1(loc_f, is_boundary)
      if (is_boundary) then
          write (*,*) i,j
      end if
    end do
  end do
  call cpu_time(finish)
  print '("T1 = ",f6.3," seconds.")',finish-start

  call cpu_time(start)
  do i = 1,M
    do j = 1,N
      loc_f%cell_face_ctr = i
      loc_f%index_p = j
      call get_boundary_status2(loc_f, is_boundary)
      if (is_boundary) then
          write (*,*) i,j
      end if
    end do
  end do
  call cpu_time(finish)
  print '("T2 = ",f6.3," seconds.")',finish-start

  contains

    subroutine get_boundary_status1(loc_f, is_boundary)
      type(face_locator), intent(in) :: loc_f
      logical, intent(out) :: is_boundary

      associate (mesh => loc_f%mesh, &
                 i => loc_f%index_p, &
                 j => loc_f%cell_face_ctr)
        is_boundary = mesh%topo%nb_indices(j, i) < 0
      end associate
    end subroutine

    subroutine get_boundary_status2(loc_f, is_boundary)
      type(face_locator), intent(in) :: loc_f
      logical, intent(out) :: is_boundary

      type(cell_locator) :: loc_p
      type(neighbour_locator) :: loc_nb

      loc_p%mesh => loc_f%mesh
      loc_p%index_p = loc_f%index_p

      loc_nb%mesh => loc_p%mesh
      loc_nb%index_p = loc_p%index_p
      loc_nb%nb_counter = loc_f%cell_face_ctr

      is_boundary = loc_nb%mesh%topo%nb_indices(loc_nb%nb_counter, loc_nb%index_p) < 0
    end subroutine

end program main
