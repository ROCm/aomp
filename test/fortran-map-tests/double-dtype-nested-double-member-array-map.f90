! Copyright © Advanced Micro Devices, Inc., or its affiliates.
!
! SPDX-License-Identifier:  MIT

program prog_a
    type :: dtype_a
     real(4) :: var_a(10)
     real(4) :: var_b
     real(4) :: var_c(10)
    end type dtype_a

    type :: dtype_b
    real(4) :: var_d
    real(4) :: var_e(10)
    real(4) :: var_f
    real(4) :: var_g(10)
    real(4) :: var_h
    type(dtype_a) :: var_i
    end type dtype_b

    type(dtype_b) :: var_j
    type(dtype_b) :: var_k
    integer :: var_l

  do var_l = 1, 10
    var_j%var_i%var_a(var_l) = var_l
    var_k%var_i%var_a(var_l) = var_l
  end do

  !$omp target map(tofrom:var_j%var_i%var_a(3:6), var_j%var_i%var_c(3:6), var_k%var_i%var_a(3:6), var_k%var_i%var_c(3:6))
    do var_l = 3, 6
      var_k%var_i%var_c(var_l) = var_j%var_i%var_a(var_l)
    end do

    do var_l = 3, 6
      var_j%var_i%var_c(var_l) = var_k%var_i%var_a(var_l)
    end do
  !$omp end target

  print*, var_j%var_i%var_c
  print*, var_k%var_i%var_a

  print*, var_k%var_i%var_c
  print*, var_j%var_i%var_a

  do var_l = 1, 2
    if (var_k%var_i%var_c(var_l) /= 0) then
      print *, "======= FORTRAN Test Failed! ======="
      stop 1
    end if
  end do

  do var_l = 3, 6
    if (var_k%var_i%var_c(var_l) /= var_l) then
      print *, "======= FORTRAN Test Failed! ======="
      stop 1
    end if
  end do

  do var_l = 7, 10
    if (var_k%var_i%var_c(var_l) /= 0) then
      print *, "======= FORTRAN Test Failed! ======="
      stop 1
    end if
  end do

  do var_l = 1, 2
    if (var_j%var_i%var_c(var_l) /= 0) then
      print *, "======= FORTRAN Test Failed! ======="
      stop 1
    end if
  end do

  do var_l = 3, 6
    if (var_j%var_i%var_c(var_l) /= var_l) then
      print *, "======= FORTRAN Test Failed! ======="
      stop 1
    end if
  end do

  do var_l = 7, 10
    if (var_j%var_i%var_c(var_l) /= 0) then
      print *, "======= FORTRAN Test Failed! ======="
      stop 1
    end if
  end do

  print*, "======= FORTRAN Test passed! ======="
end program prog_a
