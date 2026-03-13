! Copyright © Advanced Micro Devices, Inc., or its affiliates.
!
! SPDX-License-Identifier:  MIT

module mod_a
    real(8), allocatable          ::   var_a(:)
    real(8)          ::   var_b(4) = (/10,20,30,40/)
    real(8)          ::   var_c = 20

    !$omp declare target(var_a)
    !$omp declare target(var_b)
    !$omp declare target(var_c)
end module mod_a

program prog_a
    use mod_a
    real(8) :: var_d

    var_c = 10.0
    allocate(var_a(10))

    var_a(5) = 40

!$omp target map(to: var_c) map(tofrom: var_d)
    var_d = var_c
!$omp end target

    print *, var_d

 if (var_d /= 20) then
    print*, "======= FORTRAN Test Failed! ======="
    stop 1
 end if

!$omp target map(to: var_b) map(tofrom: var_d)
   var_d = var_b(3)
!$omp end target

   print *, var_d

 if (var_d /= 30) then
    print*, "======= FORTRAN Test Failed! ======="
    stop 1
 end if

!$omp target map(to: var_a) map(tofrom: var_d)
    var_d = var_a(5)
!$omp end target

  print *, var_d

if (var_d /= 40) then
    print*, "======= FORTRAN Test Failed! ======="
    stop 1
end if

print*, "======= FORTRAN Test Passed! ======="

end program
