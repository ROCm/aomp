! CLASS = A
!
!
!  This file is generated automatically by the setparams utility.
!  It sets the number of processors and the class of the NPB
!  in this directory. Do not modify it by hand.
!
    integer problem_size, niter_default
parameter (problem_size=64, niter_default=200)
    double precision dt_default
parameter (dt_default = 0.0008d0)
    logical  convertdouble
parameter (convertdouble = .false.)
    character compiletime*11
    parameter (compiletime='09 Dec 2021')
    character npbversion*5
    parameter (npbversion='3.4.1')
    character cs1*28
    parameter (cs1='/opt/rocm-4.3.1/bin/amdflang')
    character cs2*5
    parameter (cs2='$(FC)')
    character cs3*6
    parameter (cs3='(none)')
    character cs4*6
    parameter (cs4='(none)')
    character cs5*46
    parameter (cs5='-O3 -g -fopenmp -fopenmp-version=50 -fopenm...')
    character cs6*9
    parameter (cs6='$(FFLAGS)')
    character cs7*6
    parameter (cs7='randi8')
