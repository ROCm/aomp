#define N 1024
program main
  use omp_lib
  INTEGER :: errors, i, nt, tl, prevThreadLimit
  INTEGER, DIMENSION(4) :: tested_num_threads = (/1, 10, 100, 10000/)
  INTEGER, DIMENSION(4) :: tested_thread_limit = (/1, 10, 100, 10000/)
  INTEGER, DIMENSION(N) :: num_threads, thread_limit
  errors = 0

  ! Testing multiple num_threads and thread_limits values from 1 to large num.
  ! The number of threads should never be larger than the thread limit
  DO nt = 1, 4
    DO tl = 1, 4
      WRITE(*, *) "Testing thread limit (",tested_thread_limit(tl), ") num_threads(", tested_num_threads(nt), ") clauses."
      DO i = 1, N
        num_threads(i) = -1
        thread_limit(i) = -1
      END DO

      !$omp target teams distribute parallel do map(tofrom:num_threads)&
      !$omp& num_threads(tested_num_threads(nt)) thread_limit(tested_thread_limit(tl))
        DO i = 1, N
          num_threads(i) = omp_get_num_threads()
          thread_limit(i) = omp_get_thread_limit()
        END DO

        prevThreadLimit = -1

        DO i = 1, N
          IF(prevThreadLimit .ne. thread_limit(i)) then
            WRITE(*, *) "  reported thread limit =", thread_limit(i)
          ENDIF
          prevThreadLimit = thread_limit(i)
          IF(thread_limit(i) .gt. tested_thread_limit(tl) .or. (thread_limit(i) .le. 0)) then
            errors = errors + 1
          ENDIF
          IF(num_threads(i) .gt. tested_thread_limit(tl)) then
            errors = errors + 1
          ENDIF
          IF(num_threads(i) .gt. tested_num_threads(nt)) then
            errors = errors + 1
          ENDIF
        END DO
    END DO
  END DO
  WRITE(*,*) "Errors: ", errors
  IF(errors .ne. 0) then  
    WRITE(*,*) "Fail"
    stop 2
  ELSE
    WRITE(*,*) "PASS"
  ENDIF
end program main

