! Copyright © Advanced Micro Devices, Inc., or its affiliates.
!
! SPDX-License-Identifier:  MIT

program prog_a
    implicit none
    complex  :: var_a(10)
    complex  :: var_b(10)
    integer :: var_c

    do var_c = 1, 10
        var_b(var_c) = (10,20)
    end do

!$omp target map(from:var_a) map(to:var_b)
    do var_c = 1, 10
        var_a(var_c) = var_b(var_c)
    end do 
!$omp end target

 do var_c = 1, 10
 if (var_a(var_c) /= (10,20)) then
    print*, "======= FORTRAN Test Failed! ======="
    stop 1
  end if
end do

 print*, "======= FORTRAN Test passed! ======="
end program prog_a
