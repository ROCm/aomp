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
  integer :: chk, exp

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
     !$OMP PARALLEL DO
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
  
  chk = 0
  DO rkm=1,10
     write(*, "(I5)", advance="no") rkm
     write(*, "(A1)", advance="no") "|"
     DO jkm=1,10
        chk = chk + array_test(rkm,jkm)
        write(*, "(I5)", advance="no") array_test(rkm,jkm)
        write(*, "(A1)", advance="no") ":"
        write(*, "(I4)", advance="no") team_num(rkm,jkm)
        write(*, "(A1)", advance="no") ":"
        write(*, "(I2)", advance="no") thread_num(rkm,jkm)
     ENDDO
     print *
  ENDDO

  exp = 2*10*10 + (10-1)*10*10
  print *, 'exp = ', exp
  print *, 'chk = ', chk
  if (chk /= exp) then
     print *, "FAIL"
     stop 1
  endif
  print *, "PASS"
  end program main
