! Copyright © Advanced Micro Devices, Inc., or its affiliates.
!
! SPDX-License-Identifier:  MIT

program prog_a
  use omp_lib
  use iso_c_binding, only: c_loc
  implicit none

  type :: dtype_a
    integer :: var_a
    integer, allocatable :: var_b(:)
  end type dtype_a

  type(dtype_a), target :: var_c
  integer :: var_d

  var_d = omp_get_default_device()
  var_c%var_a = 10
  allocate(var_c%var_b(var_c%var_a))
  var_c%var_b = 1

  print *, "=== STORAGE Bug Reproducer (OpenMP 6.0) ==="
  print *, ""

  ! Initial state - nothing on device
  print *, "Initial:"
  call sub_a(var_c, var_d, .false.)

  !$omp target enter data map(to: var_c)
  !$omp target enter data map(to: var_c%var_b)
  print *, "After ENTER DATA (TO):"
  call sub_a(var_c, var_d, .false.)

  !$omp target exit data map(storage: var_c%var_b)
  !$omp target exit data map(storage: var_c)
  print *, "After EXIT DATA (STORAGE):"
  call sub_a(var_c, var_d, .true.)
  print *, ""

  print*, "======= FORTRAN Test passed! ======="

  print *, "If var_c:T above, the bug is present."

  deallocate(var_c%var_b)

contains

  subroutine sub_a(var_e, var_f, var_g)
    type(dtype_a), target :: var_e
    logical :: var_g
    integer :: var_f
    print '(A,L1,A,L1)', "  var_c:", func_a(var_e%var_a,var_f), "  var_c%var_b:", func_a(var_e%var_b(1),var_f)

    if ( var_g ) then
      if (func_a(var_e%var_a, var_f) .NEQV. .false.) then
        print*, "======= FORTRAN Test failed! ======="
        stop 1
      endif
    endif
  end subroutine

  logical function func_a(var_h, var_f)
    integer, target :: var_h
    integer :: var_f
    func_a = omp_target_is_present(c_loc(var_h), var_f) == 1
  end function

end program
