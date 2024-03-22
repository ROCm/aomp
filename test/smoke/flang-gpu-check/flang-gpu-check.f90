program gpucheck
    use omp_lib

    integer :: ndev
    integer :: nteams
    integer :: nthreads
    integer :: threadlmt
    logical :: on_host
    integer :: i

    ndev = omp_get_num_devices()
    write (*,'(A,I0)') 'Number of devices: ', ndev

    do i = 0,ndev-1
        call omp_set_default_device(i)
!$omp target map(from:on_host) map(from:nteams) &
!$omp&       map(from:nthreads) map(from:threadlmt)
!$omp teams
!$omp parallel
!$omp master
        if (omp_get_team_num().eq.0) then
            on_host = omp_is_initial_device()
            nteams = omp_get_num_teams()
            nthreads = omp_get_num_threads()
            threadlmt = omp_get_thread_limit()
        endif
!$omp end master
!$omp end parallel
!$omp end teams
!$omp end target
        write (*, '(A,I0,L,A,I0,A,I0,A,I0)') 'ran on GPU: ', i, .not.on_host, &
                                          ', ', nteams, &
                                          ' teams, limit of ', threadlmt, &
                                          ' threads, threads active ', nthreads
    end do
end program

