! Copyright © Advanced Micro Devices, Inc., or its affiliates.
!
! SPDX-License-Identifier:  MIT

module mod_a
    integer :: var_a(1)
!$omp declare target link(var_a)
end module mod_a

program prog_a
    use mod_a
    implicit none
    integer :: var_b(1)  
    integer :: var_c(1)  
    var_a(1) = 10
    var_b(1) = 20
   
!$omp target map(tofrom:var_a,var_b,var_c)
    var_c(1) = var_a(1) + var_b(1)
!$omp end target

if ( var_c(1) /= 30 ) then
    print*, "======= FORTRAN Test Failed! ======="
    stop 1  
end if

print*, "======= FORTRAN Test passed! ======="

end program prog_a
