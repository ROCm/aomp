program test
    implicit none
    integer :: A, threads
    threads = 128
    A = 0
    !$omp target parallel num_threads(threads)
    !$omp atomic
    A =  A + 1
    !$omp end target parallel
    print *, threads, A
    if (A  .EQ. threads) then
        print *, "PASS"
    else
        print *, "FAIL"
    end if

end program test
