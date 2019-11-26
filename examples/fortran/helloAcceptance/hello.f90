         program simple
         use omp_lib, ONLY : omp_is_initial_device
         integer onGPU
         integer n
         n=1
         print *, "Putting FORTRAN print before target GPU test!"
         onGPU = 0;
!$omp target teams distribute parallel do  map(tofrom:onGPU) private(i)
         do i=1,1
         if (.not. omp_is_initial_device()) then
              onGPU = 1
         else
              onGPU = 2
         end if
         end do
!$omp end target teams distribute parallel do
         if (onGPU .eq. 1) then
            print *, 'Fortran is running on the GPU, returned', onGPU
         else
            print *, 'Fortran is NOT running on the GPU, returned', onGPU
         end if
         print *, 'Exiting program for fortran'
         end

