       program getit
       use omp_lib
       integer num
       !$OMP TARGET
       num = omp_get_default_device()
       !$OMP END TARGET
       print *, 'num=', num  
       num = omp_get_default_device()
       print *, 'num=', num  
       end
