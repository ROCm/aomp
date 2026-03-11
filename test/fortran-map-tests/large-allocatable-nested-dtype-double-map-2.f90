! Copyright © Advanced Micro Devices, Inc., or its affiliates.
!
! SPDX-License-Identifier:  MIT

program prog_a
    type :: dtype_a
    real(4) :: var_a
    real(4) :: var_b
    real(4), allocatable :: var_c
    end type dtype_a

    type :: dtype_b
      integer(4), allocatable :: var_d
      integer(4) :: var_e
      integer(4) :: var_f
    end type dtype_b

    type :: dtype_c
     real(4) :: var_g(10)
     real(4) :: var_h
     real(4), allocatable :: var_i
     real(4) :: var_j(10)
     type(dtype_a), allocatable :: var_k
     integer(4), allocatable :: var_l
     type(dtype_b), allocatable :: var_m
    end type dtype_c

    type :: dtype_d
    real(4) :: var_n
    integer(4) :: var_o(10)
    real(4) :: var_p
    integer, allocatable :: var_q(:)
    integer(4) :: var_r
    type(dtype_c), allocatable :: var_s
    end type dtype_d

    type(dtype_d), allocatable :: var_t

    allocate(var_t)
    allocate(var_t%var_s)
    allocate(var_t%var_s%var_k)
    allocate(var_t%var_s%var_m)
    allocate(var_t%var_s%var_k%var_c)
    allocate(var_t%var_s%var_m%var_d)

!$omp target map(tofrom: var_t%var_s%var_k%var_c, var_t%var_s%var_m%var_d)
    var_t%var_s%var_k%var_c = 54
    var_t%var_s%var_m%var_d = 20
!$omp end target

  print *, var_t%var_s%var_m%var_d
  print *, var_t%var_s%var_k%var_c

  if (var_t%var_s%var_k%var_c /= 54) then
    print*, "======= FORTRAN Test Failed! ======="
    stop 1
  end if

  if (var_t%var_s%var_m%var_d /= 20) then
    print*, "======= FORTRAN Test Failed! ======="
    stop 1
  end if

  print*, "======= FORTRAN test passed! ======="
end program prog_a
