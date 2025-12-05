MODULE rep2
    implicit none
    public :: nvars

    INTEGER :: nvars = 5
END MODULE

MODULE rep

    use rep2, ONLY:nvars
    implicit none

    public :: foo

  contains

  SUBROUTINE foo()
    implicit none
    INTEGER :: i
    REAL(KIND=8) :: mm(5)
    !$omp target teams distribute parallel do private(mm)
    DO i =1,nvars
       mm(i) = 2.0_8
    END DO
    WRITE(*,*) "success"
    END SUBROUTINE
END MODULE

PROGRAM rep_arraysyntax
    use :: rep, ONLY:foo
    implicit none
    CALL foo()

END PROGRAM
