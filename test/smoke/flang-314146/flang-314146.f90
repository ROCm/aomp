      program simple
         use omp_lib, ONLY : omp_is_initial_device
         integer onGPU
         print *, "Putting print before target GPU test!"

         onGPU = 0;
!$omp target map(from:onGPU)
         if (.not. omp_is_initial_device()) then
            onGPU = 1
         else
            onGPU = 2
         end if
!$omp end target
         if (onGPU .eq. 1) then
            print *, 'Fortran is running on the GPU', onGPU
         else
            print *, 'Fortran is NOT running on the GPU', onGPU
            stop 2
         end if


         print *, 'Exiting program for fortran'
         end
