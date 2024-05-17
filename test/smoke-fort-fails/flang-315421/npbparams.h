! CLASS = A
!  
!  
!  This file is generated automatically by the setparams utility.
!  It sets the number of processors and the class of the NPB
!  in this directory. Do not modify it by hand.
!  
        integer nx, ny, nz, maxdim, niter_default
        parameter (nx=256, ny=256, nz=128, maxdim=256)
        parameter (niter_default=6)
        integer kind2
        parameter (kind2=4)
        logical  convertdouble
        parameter (convertdouble = .false.)
        character compiletime*11
        parameter (compiletime='11 Jan 2022')
        character npbversion*5
        parameter (npbversion='3.4.1')
        character cs1*24
        parameter (cs1='/opt/rocm/llvm/bin/flang')
        character cs2*5
        parameter (cs2='$(FC)')
        character cs3*6
        parameter (cs3='(none)')
        character cs4*6
        parameter (cs4='(none)')
        character cs5*46
        parameter (cs5='-O3 -fopenmp -fopenmp-targets=amdgcn-amd-am...')
        character cs6*9
        parameter (cs6='$(FFLAGS)')
        character cs7*6
        parameter (cs7='randi8')
