program test
    use omp_lib
    integer ::  n
    n = omp_get_nested()
    write(*, *) n 
    call omp_set_nested(.true.)
    n = omp_get_supported_active_levels()
    write(*, *) n 

end program test
