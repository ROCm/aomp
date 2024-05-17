program main
    use omp_lib

    integer :: nteams, nthreads

    call omp_set_num_threads(16)
    !$omp target map(nteams, nthreads)
        nteams = omp_get_num_teams()
        nthreads = omp_get_num_threads()
    !$omp end target

    write(*,*) nteams, nthreads
    if (nthreads .ne. 1) then
      write (*,*) 'wrong number of threads for 1 team no paralle'
      stop 2
      endif 
end program main

