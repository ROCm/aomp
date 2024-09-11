MODULE definitions_module
  implicit NONE  
END MODULE definitions_module

program main

  use definitions_module
  use omp_lib

  IMPLICIT NONE

  integer :: array_test(10,10)
  integer :: thread_num(10,10)
  integer :: team_num(10,10)
  integer :: jkm, rkm

  jkm = 1
  rkm = 1

  !-----------------------------------------------------------------------------
  !
  !-----------------------------------------------------------------------------
  DO rkm=1,10
     DO jkm=1,10
        array_test(rkm,jkm) = 999
        team_num(rkm,jkm) = -1
        thread_num(rkm,jkm) = -1
     ENDDO
  ENDDO

  !-----------------------------------------------------------------------------
  !
  !-----------------------------------------------------------------------------
  !$OMP TARGET TEAMS DISTRIBUTE
  DO rkm=1,10
     DO jkm=1,10
        team_num(rkm,jkm) = omp_get_team_num()
        thread_num(rkm,jkm) = omp_get_thread_num()
        array_test(rkm,jkm) = rkm + jkm
     ENDDO
  ENDDO
  !$OMP END TARGET TEAMS DISTRIBUTE
  
  !-----------------------------------------------------------------------------
  !
  !-----------------------------------------------------------------------------
  write(*, "(A6)", advance="no") "      "
  DO jkm=1,10
     write(*, "(I13)", advance="no") jkm
  ENDDO
  print *
  write(*, "(A6)", advance="no") "      "
  DO jkm=1,10
     write(*, "(A13)", advance="no") "-----------"
  ENDDO
  print *
  
  DO rkm=1,10
     write(*, "(I5)", advance="no") rkm
     write(*, "(A1)", advance="no") "|"
     DO jkm=1,10
        write(*, "(I5)", advance="no") array_test(rkm,jkm)
        write(*, "(A1)", advance="no") ":"
        write(*, "(I4)", advance="no") team_num(rkm,jkm)
        write(*, "(A1)", advance="no") ":"
        write(*, "(I2)", advance="no") thread_num(rkm,jkm)
     ENDDO
     print *
  ENDDO

  DO rkm=1,10
     DO jkm=1,10
        ! Assert that there was only one thread per team
        IF (thread_num(rkm,jkm) .ne. 0) THEN
           print *, "FAIL: Expected only one team"
           stop 1
        ENDIF
        ! Assert that each team executed one inner loop
        IF (team_num(rkm,jkm) .ne. (rkm - 1)) THEN
           print *, "FAIL: Wrong team worksharing"
           stop 1
        ENDIF
        IF (array_test(rkm,jkm) .ne. (rkm + jkm)) THEN
           print *, "FAIL: Wrong output value"
           stop 1
        ENDIF
     ENDDO
  ENDDO

  end program main
