program main

   integer x, thread, numthreads
   x = 0
   !$omp parallel num_threads(2)
     thread = omp_get_thread_num()
     if (thread .eq. 0) then
       numthreads = omp_get_num_threads()
     endif
   !$omp atomic hint(omp_sync_hint_uncontended)
     x = x + 1
   !$omp end atomic
   !$omp end parallel
   print*,x  !! 2
   if (x .ne. 2) then
     call exit(1)
   endif

end program
