PROGRAM test_target_defaultmap
    USE iso_fortran_env
    implicit none

    LOGICAL:: firstprivateCheck(10)
    COMPLEX :: scalar_complex

    scalar_complex = (1, 2)
    firstprivateCheck(10) = 0

    ! Map the same array to multiple devices. initialize with device number
    !$omp target defaultmap(tofrom: scalar) map(tofrom: firstprivateCheck)
    firstprivateCheck(10) = scalar_complex == (1, 2)
    !$omp end target

    print *, scalar_complex
    print *, firstprivateCheck(10)

    if (.not. firstprivateCheck(10)) print *, "FAILED"
    if (firstprivateCheck(10)) print *, "PASSED"


    scalar_complex = (3, 4)
    firstprivateCheck(10) = 0

    ! Map the scalar variables without defaultmap
    !$omp target map(tofrom: firstprivateCheck)
    ! checking for firstprivate copy
    firstprivateCheck(10) = scalar_complex == (3, 4)

    ! Attempting value change
    scalar_complex = (5, 5)
    !$omp end target

    print *, scalar_complex
    print *, firstprivateCheck(10)

    if (.not. firstprivateCheck(10)) print *, "FAILED"
    if (firstprivateCheck(10)) print *, "PASSED"

END PROGRAM test_target_defaultmap
