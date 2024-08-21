      MODULE testmod
        IMPLICIT NONE

        PRIVATE
        SAVE

        PUBLIC :: test

        CONTAINS

                SUBROUTINE test(a,b)
                  INTEGER,INTENT(IN) :: a
                  LOGICAL,OPTIONAL,INTENT(IN) :: b

                  !$OMP PARALLEL SHARED(b)
                  IF(PRESENT(b)) THEN
                          WRITE(*,*) "b present"
                  ELSE
                          WRITE(*,*) "b not present"
                  END IF
                  !$OMP END PARALLEL
                END SUBROUTINE
      END MODULE testmod


      PROGRAM test_optional
              USE :: testmod
              INTEGER :: a
              LOGICAL :: b

              CALL test(a,b)

      END PROGRAM
