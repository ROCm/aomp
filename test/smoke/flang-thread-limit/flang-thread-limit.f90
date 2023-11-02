program main
 USE omp_lib
 INTEGER, DIMENSION(1024) :: a = (/(1, i=0,1024 - 1)/)
 INTEGER, DIMENSION(1024) :: a_ref = (/(1, i=0,1024 - 1)/)
 INTEGER, DIMENSION(1024) :: b = (/(i, i=0,1024 - 1)/)
 INTEGER, DIMENSION(1024) :: c = (/(2 * i, i=0,1024 - 1)/)
 INTEGER :: OMPVV_NUM_TEAMS_DEVICE = 8
 INTEGER :: OMPVV_NUM_THREADS_DEVICE = 8
 INTEGER :: num_teams = 0
 INTEGER :: num_threads = 0
 INTEGER :: i

 DO i = 1, 1024
  a_ref(i) = a_ref(i) + b(i) * c(i);
 END DO

 !$omp target teams distribute parallel do  map(from:num_teams, num_threads) &
 !$omp& num_teams(OMPVV_NUM_TEAMS_DEVICE)  num_threads(OMPVV_NUM_THREADS_DEVICE)
 DO i = 1, 1024
  a(i) = a(i) + b(i) * c(i);
  num_teams = omp_get_num_teams();
  num_threads = omp_get_num_threads();
 END DO

 PRINT *, "Actual number of threads"
 PRINT *, num_threads
 PRINT *, "Actual number of teams"
 PRINT *, num_teams

 DO i = 1, 1024
  IF (a(i) .NE. a_ref(i)) THEN
   PRINT *, "Array mismatch"
   STOP 2
  ENDIF
 END DO
 PRINT *, "Passed"
 return
end program main
