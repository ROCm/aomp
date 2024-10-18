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
    print *, "Hello OpenMP", ival
    print *, ival1, ival2, ival4, ival8
    print *, fval, dval, cfval, cdval
!$omp end target
end
