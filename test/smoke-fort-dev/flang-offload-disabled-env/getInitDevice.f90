       program test_offload_disabled_env
       use omp_lib
       integer num, initial_dev
       
       ! Test that omp_get_default_device returns the initial device
       ! when OMP_TARGET_OFFLOAD=DISABLED, even when OMP_DEFAULT_DEVICE=2
       
       ! Get initial device (should be 0 when offload is disabled)
       initial_dev = omp_get_initial_device()
       print *, 'initial device =', initial_dev
       
       ! Get default device (should also be 0, not 2, when offload is disabled)
       num = omp_get_default_device()
       print *, 'default device =', num
       
       ! Call from within target region
       !$OMP TARGET
       num = omp_get_default_device()
       !$OMP END TARGET
       print *, 'default device from target =', num
       
       ! Verify they match
       if (num .eq. initial_dev) then
           print *, 'PASS: default device equals initial device'
       else
           print *, 'FAIL: default device does not equal initial device'
           call exit(1)
       end if
       
       end
