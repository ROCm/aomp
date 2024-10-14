program main
    implicit none
    integer         :: ival = -1
    integer(kind=1) :: ival1 = 1
    integer(kind=2) :: ival2 = 2
    integer(kind=4) :: ival4 = 4
    integer(kind=8) :: ival8 = 8
    real(kind=4)    :: fval = 14
    real(kind=8)    :: dval = 18
    complex(kind=4) :: cfval = (24, 25)
    complex(kind=8) :: cdval = (28, 29)
!$omp target
    write(*,'(a, i10)') "Hello OpenMP", ival
    write(*,*) ival1, ival2, ival4, ival8
    write(*,10) ival1, ival2, ival4, ival8
    write(*,20) ival1, ival2, ival4, ival8
    write(*,*) fval, dval, cfval, cdval
10 format(i5, i5, i5, i5)
!$omp end target
20 format(i3, i3, i3, i3)
end
