program main  
        use iso_fortran_env, only: output_unit
        implicit none
        integer :: hostArray(10), chk = 0, calltarget
        external hello
        integer :: hello
        chk = calltarget(10, 2)
        print *, 'chk    =', chk
        call flush(output_unit)
        call hello(chk, 3);

        if ( chk /= 20 ) then
                print*, "======= FORTRAN Test failed! ======="
                stop 1  
        end if
end program main

function calltarget(chkval, mult) result(chktgt)
        use iso_fortran_env, only: output_unit
        implicit none
        integer :: chkval, mult, chktgt
        external hello_dev
        integer :: hello_dev
        print *, 'chkval =', chkval
        call flush(output_unit)
!$omp target map(from:chktgt)
        chktgt = hello_dev(chkval, mult)
!$omp end target
        call flush(output_unit)
        print *, 'chktgt =', chktgt
end function calltarget
