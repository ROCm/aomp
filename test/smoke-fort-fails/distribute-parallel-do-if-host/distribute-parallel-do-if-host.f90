program main
  implicit none
  integer, parameter :: nteams = 2, nthreads = 3, n = nteams * nthreads
  integer :: i, idx(n), teams(n), threads(n)

  call init("teams + parallel")
  !$omp teams num_teams(nteams) if(.true.)
  !$omp distribute parallel do num_threads(nthreads) if(.true.)
  do i=1,n
    call kernel(i)
  end do
  !$omp end teams

  if (.not. verify_idx()) then
    print *, "idx:", idx
    call exit(1)
  endif

  if (.not. verify_teams_parallel()) then
    print *, "teams:", teams
    print *, "threads:", threads
    call exit(1)
  endif
  print *, "succeeded"

  call init("teams")
  !$omp teams num_teams(nteams) if(.true.)
  !$omp distribute parallel do num_threads(nthreads) if(.false.)
  do i=1,n
    call kernel(i)
  end do
  !$omp end teams

  if (.not. verify_idx()) then
    print *, "idx:", idx
    call exit(1)
  endif

  if (.not. verify_teams()) then
    print *, "teams:", teams
    print *, "threads:", threads
    call exit(1)
  endif
  print *, "succeeded"

  call init("parallel")
  !$omp teams num_teams(nteams) if(.false.)
  !$omp distribute parallel do num_threads(nthreads) if(.true.)
  do i=1,n
    call kernel(i)
  end do
  !$omp end teams

  if (.not. verify_idx()) then
    print *, "idx:", idx
    call exit(1)
  endif

  if (.not. verify_parallel()) then
    print *, "teams:", teams
    print *, "threads:", threads
    call exit(1)
  endif
  print *, "succeeded"

contains
  subroutine init(name)
    character(*), intent(in) :: name
    print *, name
    do i=1,n
      idx(i) = 999
      teams(i) = 999
      threads(i) = 999
    end do
  end subroutine

  subroutine kernel(i)
    use omp_lib
    integer, intent(in) :: i

    idx(i) = i
    teams(i) = omp_get_team_num()
    threads(i) = omp_get_thread_num()
  end subroutine

  function verify_idx()
    logical :: verify_idx
    verify_idx = .true.
    do i=1,n
      if (idx(i) .ne. i) then
        print *, "index(", i, "):", idx(i), "not equal to", i
        verify_idx = .false.
        return
      endif
    end do
  end function

  function verify_teams_parallel()
    logical :: verify_teams_parallel
    integer :: team, thread
    verify_teams_parallel = .true.
    i = 1
    do team=1,nteams
      do thread=1,nthreads
        if (teams(i) .ne. team - 1) then
          print *, "teams(", i, "):", teams(i), "not equal to", team - 1
          verify_teams_parallel = .false.
          return
        endif
        if (threads(i) .ne. thread - 1) then
          print *, "threads(", i, "):", threads(i), "not equal to", thread - 1
          verify_teams_parallel = .false.
          return
        endif
        i = i + 1
      end do
    end do
  end function

  function verify_teams()
    logical :: verify_teams
    verify_teams = .true.
    do i=1,n
      if (teams(i) .ne. 0) then
        print *, "teams(", i, "):", teams(i), "not equal to 0"
        verify_teams = .false.
        return
      endif
      if (threads(i) .ne. 0) then
        print *, "threads(", i, "):", threads(i), "not equal to 0"
        verify_teams = .false.
        return
      endif
    end do
  end function

  function verify_parallel()
    logical :: verify_parallel
    integer :: team, thread
    verify_parallel = .true.
    i = 1
    do team=1,nteams
      do thread=1,nthreads
        if (teams(i) .ne. 0) then
          print *, "teams(", i, "):", teams(i), "not equal to 0"
          verify_parallel = .false.
          return
        endif
        if (threads(i) .ne. (i-1) / nteams) then
          print *, "threads(", i, "):", threads(i), "not equal to", (i-1) / nteams
          verify_parallel = .false.
          return
        endif
        i = i + 1
      end do
    end do
  end function
end program
