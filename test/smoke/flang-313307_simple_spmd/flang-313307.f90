program main
    use omp_lib
    integer :: nteams, nthreads

    !$omp target parallel map(nteams, nthreads)
        nteams = omp_get_num_teams()
        nthreads = omp_get_num_threads()
    !$omp end target parallel

    write(*,*) nteams, nthreads
    if (nthreads .eq. 1) then
      write (*,*) "Failed nthreads for spmd kernell"
      stop 2
    endif
end program main
