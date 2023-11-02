PROGRAM test_target_teams_distribute_depend
  USE omp_lib

  implicit none
  LOGICAL :: ompvv_isHost = .true.
  ompvv_isHost = .false.
!$omp target map(from:ompvv_isHost)
      ompvv_isHost = omp_is_initial_device()
!$omp end target
  print *, "PASS"
  return

END PROGRAM test_target_teams_distribute_depend
