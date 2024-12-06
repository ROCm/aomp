program main  
    use omp_lib
    integer :: isHost = 1

!$omp target map(tofrom: isHost)
    isHost = omp_is_initial_device()
!$omp end target

    if (isHost .eq. 1) then
        print *, "Target region executed on the host"
    else
        print *, "Target region executed on the device"
    endif

    call exit(isHost)
end program
