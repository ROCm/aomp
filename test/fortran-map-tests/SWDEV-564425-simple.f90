! Copyright © Advanced Micro Devices, Inc., or its affiliates.
!
! SPDX-License-Identifier:  MIT

program prog_a
  use omp_lib
  use iso_c_binding
  implicit none

  real(kind=8), pointer :: var_a(:)

  integer :: var_b, var_c, var_d
  logical var_e

  var_b = 500000

  allocate(var_a(var_b))
  var_a = [(real(var_c,kind=8), var_c=1,var_b)]

  print *, "var_a    =", var_a(1:10)
  print *, size(var_a)

  !$omp target enter data map(to: var_a)
  var_e = (omp_target_is_present(C_LOC(var_a), omp_get_default_device()) /= 0)
  print *, "before parallel do var_a is present on device: ", var_e
  !$omp target teams distribute parallel do map(present: var_a)
  do var_d = 1, size(var_a)
    var_a(var_d) = var_a(var_d) + 100.0d0
  end do

  !$omp target update from(var_a)

  if(omp_target_is_present(C_LOC(var_a), omp_get_default_device()) == 0) then
    print *, "Before release: var_a NOT on device"
  else
    print *, "Before release: var_a on device"
  endif

  !$omp target exit data map(ref_ptr, storage: var_a)

  var_e = omp_target_is_present(C_LOC(var_a), omp_get_default_device()) /= 0
  print *, "After ref_ptr release: var_a PRESENT on device ", var_e

  if (var_e .NEQV. .true.) then
      print*, "======= FORTRAN Test Failed! ======="
      stop 1
   end if

  !$omp target exit data map(ref_ptee, storage: var_a)

  var_e = omp_target_is_present(C_LOC(var_a), omp_get_default_device()) /= 0
  print *, "After ref_ptee release: var_a PRESENT on device ", var_e

  if (var_e .NEQV. .false.) then
      print*, "======= FORTRAN Test Failed! ======="
      stop 1
   end if

  deallocate(var_a)

  print*, "======= FORTRAN Test Passed! ======="
end program
