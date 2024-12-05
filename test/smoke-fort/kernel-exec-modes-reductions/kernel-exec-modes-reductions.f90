subroutine validate(acc, expected, name)
  implicit none
  integer, intent(in) :: acc, expected
  character(len = *), intent(in) :: name

  if (acc .ne. expected) then
    write(*,*) name, acc
    call exit(1)
  endif
end subroutine validate

program kernel_exec_modes_reductions
  use omp_lib
  implicit none
  integer :: i, j
  integer, parameter :: teams=5, threads=10
  integer :: outer_iters, inner_iters, acc

  outer_iters = 50
  inner_iters = 100

  ! Combined SPMD
  acc = 0
  !$omp target teams distribute parallel do num_teams(teams) thread_limit(threads) collapse(2) reduction(+:acc)
  do i=1,outer_iters
    do j=1,inner_iters
      acc = acc + 1
    end do
  end do
  call validate(acc, outer_iters * inner_iters, "Combined SPMD")

  ! Split SPMD
  acc = 0
  !$omp target teams num_teams(teams) thread_limit(threads) reduction(+:acc)
  !$omp distribute parallel do collapse(2) reduction(+:acc)
  do i=1,outer_iters
    do j=1,inner_iters
      acc = acc + 1
    end do
  end do
  !$omp end target teams
  call validate(acc, outer_iters * inner_iters, "Split SPMD")

  ! Combined Generic-SPMD
  acc = 0
  !$omp target teams distribute num_teams(teams) thread_limit(threads) reduction(+:acc)
  do i=1,outer_iters
    !$omp parallel do reduction(+:acc)
    do j=1,inner_iters
      acc = acc + 1
    end do
  end do
  call validate(acc, outer_iters * inner_iters, "Combined Generic-SPMD")

  ! Split Generic-SPMD
  acc = 0
  !$omp target teams num_teams(teams) thread_limit(threads) reduction(+:acc)
  !$omp distribute
  do i=1,outer_iters
    !$omp parallel do reduction(+:acc)
    do j=1,inner_iters
      acc = acc + 1
    end do
  end do
  !$omp end target teams
  call validate(acc, outer_iters * inner_iters, "Split Generic-SPMD")

  ! Generic
  acc = 0
  !$omp target teams num_teams(teams) thread_limit(threads) reduction(+:acc)
  !$omp parallel do reduction(+:acc) collapse(2)
  do i=1,outer_iters
    do j=1,inner_iters
      acc = acc + 1
    end do
  end do
  !$omp end parallel do
  !$omp end target teams
  call validate(acc, teams * outer_iters * inner_iters, "Generic")
end program kernel_exec_modes_reductions
