subroutine inner(n)
        integer :: n


     !$OMP target device(n)
            n = 1 
     !$OMP end target
     !$OMP target device(device_num: n)
            n = 2
     !$OMP end target
     !$OMP target device(ancestor: n)
            n = 3 
     !$OMP end target
end subroutine inner

program loop_test

     call inner(2)
end program loop_test

