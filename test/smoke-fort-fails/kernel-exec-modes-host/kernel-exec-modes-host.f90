subroutine init_arr(v, cols, rows)
  implicit none
  integer, intent(in) :: cols, rows
  integer, intent(inout) :: v(cols, rows)
  integer :: i, j

  do i=1,rows
    do j=1,cols
      v(j, i) = 999
    end do
  end do
end subroutine init_arr

subroutine validate(v, cols, rows, name)
  implicit none
  integer, intent(in) :: cols, rows
  integer, intent(inout) :: v(cols, rows)
  character(len = *), intent(in) :: name
  integer :: i, j

  do i=1,rows
    do j=1,cols
      if (v(j, i) .ne. (i-1) * cols + (j-1)) then
        write(*,*) name, v(:, :)
        call exit(1)
      endif
    end do
  end do
end subroutine validate

program kernel_exec_modes_host
  use omp_lib
  implicit none
  integer :: i, j
  integer, parameter :: teams=5, threads=10
  integer :: a(threads, teams)

  ! Combined SPMD
  call init_arr(a, threads, teams)
  !$omp target teams distribute parallel do num_teams(teams) thread_limit(threads) collapse(2)
  do i=1,teams
    do j=1,threads
      a(j, i) = omp_get_team_num() * omp_get_num_threads() + omp_get_thread_num()
    end do
  end do
  call validate(a, threads, teams, "Combined SPMD")

  ! Split SPMD
  call init_arr(a, threads, teams)
  !$omp target teams num_teams(teams) thread_limit(threads)
  !$omp distribute parallel do collapse(2)
  do i=1,teams
    do j=1,threads
      a(j, i) = omp_get_team_num() * omp_get_num_threads() + omp_get_thread_num()
    end do
  end do
  !$omp end target teams
  call validate(a, threads, teams, "Split SPMD")

  ! Combined Generic-SPMD
  call init_arr(a, threads, teams)
  !$omp target teams distribute num_teams(teams) thread_limit(threads)
  do i=1,teams
    !$omp parallel do
    do j=1,threads
      a(j, i) = omp_get_team_num() * omp_get_num_threads() + omp_get_thread_num()
    end do
  end do
  call validate(a, threads, teams, "Combined Generic-SPMD")

  ! Split Generic-SPMD
  call init_arr(a, threads, teams)
  !$omp target teams num_teams(teams) thread_limit(threads)
  !$omp distribute
  do i=1,teams
    !$omp parallel do
    do j=1,threads
      a(j, i) = omp_get_team_num() * omp_get_num_threads() + omp_get_thread_num()
    end do
  end do
  !$omp end target teams
  call validate(a, threads, teams, "Split Generic-SPMD")

  ! Generic
  call init_arr(a, threads, teams)
  !$omp target teams num_teams(teams) thread_limit(threads)
  !$omp parallel
  a(omp_get_thread_num() + 1, omp_get_team_num() + 1) = omp_get_team_num() * omp_get_num_threads() + omp_get_thread_num()
  !$omp end parallel
  !$omp end target teams
  call validate(a, threads, teams, "Generic")
end program kernel_exec_modes_host
