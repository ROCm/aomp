! Copyright © Advanced Micro Devices, Inc., or its affiliates.
!
! SPDX-License-Identifier:  MIT

  subroutine sub_a(var_a)
      implicit none
      integer, intent(in), dimension(10) :: var_a
      integer var_b
      do var_b = 1, 10
            PRINT *, var_a(var_b)
      end do
  end subroutine

  program prog_a
      implicit none
      integer :: var_a(10) = (/0,0,0,0,0,0,0,0,0,0/)
      integer :: var_b

    !$omp target map(tofrom:var_a)
       var_a(1) = 20
       var_a(5) = 10
    !$omp end target

    call sub_a(var_a)

    if (var_a(1) /= 20) then
      print*, "======= FORTRAN Test Failed! ======="
      stop 1
    end if

   if (var_a(5) /= 10) then
     print*, "======= FORTRAN Test Failed! ======="
     stop 1
   end if

  do var_b = 2, 4
    if (var_a(var_b) /= 0) then
        print*, "======= FORTRAN Test Failed! ======="
        stop 1
    end if
  end do

  do var_b = 6, 10
    if (var_a(var_b) /= 0) then
        print*, "======= FORTRAN Test Failed! ======="
        stop 1
    end if
  end do

  print*, "======= FORTRAN Test passed! ======="
end program
