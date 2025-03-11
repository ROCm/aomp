program main

  use omp_lib

  IMPLICIT NONE

  integer :: count1, count2, rkm
  integer :: counts_team


  count1 = 0
  !$OMP TARGET TEAMS MAP(tofrom: count1) PRIVATE(counts_team, rkm)
       counts_team = 0
       !$OMP PARALLEL
            !$OMP DO 
            DO rkm = 1,4
                 !$OMP ATOMIC
                 counts_team = counts_team + 1
            END DO
       !$OMP END PARALLEL
       IF (omp_get_team_num() .eq. 0 ) THEN
           count1 = counts_team
       END IF
  !$OMP END TARGET TEAMS
  if (count1.ne. 4) then
     print *, "FAIL COUNT 1"
     stop 1
  endif

  count2 = 0
  !$OMP TARGET TEAMS MAP(tofrom: count1) PRIVATE(counts_team, rkm)
       counts_team = 0
       !$OMP PARALLEL
            !$OMP DO REDUCTION(+:counts_team)
            DO rkm = 1,4
                 counts_team = counts_team + 1
            END DO
       !$OMP END PARALLEL
       IF (omp_get_team_num() .eq. 0 ) THEN
           count2 = counts_team
       END IF
  !$OMP END TARGET TEAMS
  if (count2.ne. 4) then
     print *, "FAIL COUNT 2"
     stop 1
  endif

end program main
