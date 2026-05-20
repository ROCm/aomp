! Copyright © Advanced Micro Devices, Inc., or its affiliates.
!
! SPDX-License-Identifier:  MIT

program prog_a
  use omp_lib
  use iso_c_binding
  implicit none

  type :: dtype_a
     real(kind=8) :: var_a
     real(kind=8), pointer :: var_b(:)
  end type dtype_a

  type(dtype_a), target :: var_c
  integer :: var_d, var_e, var_f
  logical var_g

  var_d = 500000
  var_c%var_a = 1.23d0

  allocate(var_c%var_b(var_d))
  var_c%var_b = [(real(var_e,kind=8), var_e=1,var_d)]

  print *, "var_a =", var_c%var_a
  print *, "var_b    =", var_c%var_b(1:10)
  print *, size(var_c%var_b)

  !$omp target enter data map(to: var_c, var_c%var_b)
  var_g = (omp_target_is_present(C_LOC(var_c%var_b), omp_get_default_device()) /= 0)
  print *, "before parallel do var_c%var_b is present on device: ", var_g
  !$omp target teams distribute parallel do map(present: var_c)
  do var_f = 1, size(var_c%var_b)
    var_c%var_b(var_f) = var_c%var_b(var_f) + 100.0d0
  end do

  !$omp target map(present: var_c)
  var_c%var_a = var_c%var_a + 10.0d0
  !$omp end target

  print *, "var_a =", var_c%var_a
  print *, "var_b    =", var_c%var_b(1:10)
  !$omp target update from(var_c)
  print *, "=== Host after device update ==="
  print *, "var_a =", var_c%var_a
  print *, "var_b    =", var_c%var_b(1:10)


  if(omp_target_is_present(C_LOC(var_c), omp_get_default_device()) == 0) then
    print *, "Before release: var_c NOT on device"
  else
    print *, "Before release: var_c on device"
  endif

  if(omp_target_is_present(C_LOC(var_c%var_b), omp_get_default_device()) == 0) then
    print *, "Before release: var_c%var_b NOT on device"
  else
    print *, "Before release: var_c%var_b on device"
  endif

  !$omp target exit data map(ref_ptee, storage: var_c%var_b)

  var_g = omp_target_is_present(C_LOC(var_c), omp_get_default_device()) /= 0
  print *, "After ref_ptee, release of var_c%var_b: var_c PRESENT on device ", var_g

  if (var_g .NEQV. .true.) then
      print*, "======= FORTRAN Test Failed! ======="
      stop 1
   end if

  var_g = omp_target_is_present(C_LOC(var_c%var_b), omp_get_default_device()) /= 0
  print *, "After ref_ptee, release of var_c%var_b: var_c%var_b PRESENT on device ", var_g

  if (var_g .NEQV. .false.) then
      print*, "======= FORTRAN Test Failed! ======="
      stop 1
   end if

  !$omp target exit data map(ref_ptr, storage: var_c%var_b)
  !$omp target exit data map(delete: var_c)

  var_g = omp_target_is_present(C_LOC(var_c), omp_get_default_device()) /= 0
  print *, "After exit delete of var_c: var_c PRESENT on device ", var_g

  if (var_g .NEQV. .false.) then
      print*, "======= FORTRAN Test Failed! ======="
      stop 1
   end if

  print*, "======= FORTRAN Test Passed! ======="
end program
