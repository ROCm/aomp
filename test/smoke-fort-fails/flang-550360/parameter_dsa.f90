!
!Copyright © Advanced Micro Devices, Inc., or its affiliates.
!
!SPDX-License-Identifier:  MIT
! 
program shrd
    implicit none
    integer, parameter :: n = 7
    real, parameter :: m = 7.0


    !$omp parallel shared(n) firstprivate(m)
        print *, n, m
    !$omp end parallel
end program shrd
