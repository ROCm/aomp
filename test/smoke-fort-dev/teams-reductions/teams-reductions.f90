subroutine host(result)
  implicit none
  logical, intent(out) :: result
  integer, parameter :: N = 30, HIGH = 4, LOW = 1
  integer :: i, j, x, y(N, N), z(N)

  result = .false.

  print *, "1"
  x = 0
  !$omp teams distribute reduction(+:x) num_teams(HIGH)
  do i=1, N*N*N
    x = x + 1
  end do

  if (x .ne. N*N*N) then
    print *, x
    return
  end if

  print *, "2"
  x = 0
  !$omp teams distribute reduction(+:x) num_teams(LOW)
  do i=1, N*N
    !$omp parallel do reduction(+:x) num_threads(HIGH)
    do j=1, N
      x = x + 1
    end do
  end do

  if (x .ne. N*N*N) then
    print *, x
    return
  end if

  print *, "3"
  x = 0
  !$omp teams distribute reduction(+:x) num_teams(HIGH)
  do i=1, N*N
    !$omp parallel do reduction(+:x) num_threads(LOW)
    do j=1, N
      x = x + 1
    end do
  end do

  if (x .ne. N*N*N) then
    print *, x
    return
  end if

  print *, "4"
  x = 0
  !$omp teams distribute reduction(+:x) num_teams(HIGH)
  do i=1, N*N
    !$omp parallel do reduction(+:x) num_threads(HIGH)
    do j=1, N
      x = x + 1
    end do
  end do

  if (x .ne. N*N*N) then
    print *, x
    return
  end if

  print *, "5"
  x = 0
  !$omp teams distribute reduction(+:x) num_teams(LOW)
  do i=1, N*N
    x = x + 1
    !$omp parallel do reduction(+:x) num_threads(HIGH)
    do j=1, N
      x = x + 1
    end do
    x = x + 1
  end do

  if (x .ne. N*N*(N+2)) then
    print *, x
    return
  end if

  print *, "6"
  x = 0
  !$omp teams distribute reduction(+:x) num_teams(HIGH)
  do i=1, N*N
    x = x + 1
    !$omp parallel do reduction(+:x) num_threads(LOW)
    do j=1, N
      x = x + 1
    end do
    x = x + 1
  end do

  if (x .ne. N*N*(N+2)) then
    print *, x
    return
  end if

  print *, "7"
  x = 0
  !$omp teams distribute reduction(+:x) num_teams(HIGH)
  do i=1, N*N
    x = x + 1
    !$omp parallel do reduction(+:x) num_threads(HIGH)
    do j=1, N
      x = x + 1
    end do
    x = x + 1
  end do

  if (x .ne. N*N*(N+2)) then
    print *, x
    return
  end if

  print *, "8"
  y(:, :) = 0
  !$omp teams distribute
  do i=1, N
    y(i, :) = y(i, :) + 1
  end do

  do i=1, N
    do j=1, N
      if (y(i, j) .ne. 1) then
        print *, y(:, :)
        return
      end if
    end do
  end do

  print *, "9"
  y(:, :) = 0
  !$omp teams distribute num_teams(LOW)
  do i=1, N
    !$omp parallel do num_threads(HIGH)
    do j=1, N
      y(i, j) = y(i, j) + 1
    end do
  end do

  do i=1, N
    do j=1, N
      if (y(i, j) .ne. 1) then
        print *, y(:, :)
        return
      end if
    end do
  end do

  print *, "10"
  y(:, :) = 0
  !$omp teams distribute num_teams(HIGH)
  do i=1, N
    !$omp parallel do num_threads(LOW)
    do j=1, N
      y(i, j) = y(i, j) + 1
    end do
  end do

  do i=1, N
    do j=1, N
      if (y(i, j) .ne. 1) then
        print *, y(:, :)
        return
      end if
    end do
  end do

  print *, "11"
  y(:, :) = 0
  !$omp teams distribute num_teams(HIGH)
  do i=1, N
    !$omp parallel do num_threads(HIGH)
    do j=1, N
      y(i, j) = y(i, j) + 1
    end do
  end do

  do i=1, N
    do j=1, N
      if (y(i, j) .ne. 1) then
        print *, y(:, :)
        return
      end if
    end do
  end do

  print *, "12"
  z(:) = 0
  !$omp teams distribute num_teams(LOW)
  do i=1, N
    !$omp parallel do num_threads(HIGH) reduction(+:z(i))
    do j=1, N
      z(i) = z(i) + 1
    end do
  end do

  do i=1, N
    if (z(i) .ne. N) then
      print *, z(:)
      return
    end if
  end do

  print *, "13"
  z(:) = 0
  !$omp teams distribute num_teams(HIGH)
  do i=1, N
    !$omp parallel do num_threads(LOW) reduction(+:z(i))
    do j=1, N
      z(i) = z(i) + 1
    end do
  end do

  do i=1, N
    if (z(i) .ne. N) then
      print *, z(:)
      return
    end if
  end do

  print *, "14"
  z(:) = 0
  !$omp teams distribute num_teams(HIGH)
  do i=1, N
    !$omp parallel do num_threads(HIGH) reduction(+:z(i))
    do j=1, N
      z(i) = z(i) + 1
    end do
  end do

  do i=1, N
    if (z(i) .ne. N) then
      print *, z(:)
      return
    end if
  end do

  result = .true.
end subroutine

subroutine device(result)
  implicit none
  logical, intent(out) :: result
  integer, parameter :: N = 30, HIGH = 4, LOW = 1
  integer :: i, j, x, y(N, N), z(N)

  result = .false.

  print *, "1"
  x = 0
  !$omp target teams distribute reduction(+:x) num_teams(HIGH)
  do i=1, N*N*N
    x = x + 1
  end do

  if (x .ne. N*N*N) then
    print *, x
    return
  end if

  print *, "2"
  x = 0
  !$omp target teams distribute reduction(+:x) num_teams(LOW) thread_limit(HIGH)
  do i=1, N*N
    !$omp parallel do reduction(+:x)
    do j=1, N
      x = x + 1
    end do
  end do

  if (x .ne. N*N*N) then
    print *, x
    return
  end if

  print *, "3"
  x = 0
  !$omp target teams distribute reduction(+:x) num_teams(HIGH) thread_limit(LOW)
  do i=1, N*N
    !$omp parallel do reduction(+:x)
    do j=1, N
      x = x + 1
    end do
  end do

  if (x .ne. N*N*N) then
    print *, x
    return
  end if

  print *, "4"
  x = 0
  !$omp target teams distribute reduction(+:x) num_teams(HIGH) thread_limit(HIGH)
  do i=1, N*N
    !$omp parallel do reduction(+:x)
    do j=1, N
      x = x + 1
    end do
  end do

  if (x .ne. N*N*N) then
    print *, x
    return
  end if

  print *, "5"
  x = 0
  !$omp target teams distribute reduction(+:x) num_teams(LOW) thread_limit(HIGH)
  do i=1, N*N
    x = x + 1
    !$omp parallel do reduction(+:x)
    do j=1, N
      x = x + 1
    end do
    x = x + 1
  end do

  if (x .ne. N*N*(N+2)) then
    print *, x
    return
  end if

  print *, "6"
  x = 0
  !$omp target teams distribute reduction(+:x) num_teams(HIGH) thread_limit(LOW)
  do i=1, N*N
    x = x + 1
    !$omp parallel do reduction(+:x)
    do j=1, N
      x = x + 1
    end do
    x = x + 1
  end do

  if (x .ne. N*N*(N+2)) then
    print *, x
    return
  end if

  print *, "7"
  x = 0
  !$omp target teams distribute reduction(+:x) num_teams(HIGH) thread_limit(HIGH)
  do i=1, N*N
    x = x + 1
    !$omp parallel do reduction(+:x)
    do j=1, N
      x = x + 1
    end do
    x = x + 1
  end do

  if (x .ne. N*N*(N+2)) then
    print *, x
    return
  end if

  print *, "8"
  y(:, :) = 0
  !$omp target teams distribute
  do i=1, N
    y(i, :) = y(i, :) + 1
  end do

  do i=1, N
    do j=1, N
      if (y(i, j) .ne. 1) then
        print *, y(:, :)
        return
      end if
    end do
  end do

  print *, "9"
  y(:, :) = 0
  !$omp target teams distribute num_teams(LOW) thread_limit(HIGH)
  do i=1, N
    !$omp parallel do
    do j=1, N
      y(i, j) = y(i, j) + 1
    end do
  end do

  do i=1, N
    do j=1, N
      if (y(i, j) .ne. 1) then
        print *, y(:, :)
        return
      end if
    end do
  end do

  print *, "10"
  y(:, :) = 0
  !$omp target teams distribute num_teams(HIGH) thread_limit(LOW)
  do i=1, N
    !$omp parallel do
    do j=1, N
      y(i, j) = y(i, j) + 1
    end do
  end do

  do i=1, N
    do j=1, N
      if (y(i, j) .ne. 1) then
        print *, y(:, :)
        return
      end if
    end do
  end do

  print *, "11"
  y(:, :) = 0
  !$omp target teams distribute num_teams(HIGH) thread_limit(HIGH)
  do i=1, N
    !$omp parallel do
    do j=1, N
      y(i, j) = y(i, j) + 1
    end do
  end do

  do i=1, N
    do j=1, N
      if (y(i, j) .ne. 1) then
        print *, y(:, :)
        return
      end if
    end do
  end do

  print *, "12"
  z(:) = 0
  !$omp target teams distribute num_teams(LOW) thread_limit(HIGH)
  do i=1, N
    !$omp parallel do reduction(+:z(i))
    do j=1, N
      z(i) = z(i) + 1
    end do
  end do

  do i=1, N
    if (z(i) .ne. N) then
      print *, z(:)
      return
    end if
  end do

  print *, "13"
  z(:) = 0
  !$omp target teams distribute num_teams(HIGH) thread_limit(LOW)
  do i=1, N
    !$omp parallel do reduction(+:z(i))
    do j=1, N
      z(i) = z(i) + 1
    end do
  end do

  do i=1, N
    if (z(i) .ne. N) then
      print *, z(:)
      return
    end if
  end do

  print *, "14"
  z(:) = 0
  !$omp target teams distribute num_teams(HIGH) thread_limit(HIGH)
  do i=1, N
    !$omp parallel do reduction(+:z(i))
    do j=1, N
      z(i) = z(i) + 1
    end do
  end do

  do i=1, N
    if (z(i) .ne. N) then
      print *, z(:)
      return
    end if
  end do

  result = .true.
end subroutine

program main
  implicit none
  logical :: result

  call host(result)
  if (.not. result) then
    print *, "Host device execution failed"
    stop 1
  end if

  call device(result)
  if (.not. result) then
    print *, "Target device execution failed"
    stop 1
  end if
end program
