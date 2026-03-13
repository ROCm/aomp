! Copyright © Advanced Micro Devices, Inc., or its affiliates.
!
! SPDX-License-Identifier:  MIT

program prog_a
  use omp_lib
  use iso_c_binding
  implicit none

  type :: dtype_a
     real(kind=8) :: var_a
     integer(kind=4), pointer :: var_b
     integer(kind=8), pointer :: var_c(:)
  end type dtype_a

  type :: dtype_b
     real(kind=8) :: var_d
     real(kind=8) :: var_e
     integer(kind=4), pointer :: var_f
     type(dtype_a) :: var_g
  end type dtype_b

  type :: dtype_c
     type(dtype_b) :: var_h
     real(kind=8) :: var_i
     real(kind=8), pointer :: var_j(:)
     type(dtype_a) :: var_k
  end type dtype_c

  type(dtype_c), pointer :: var_l
  integer :: var_m, var_n, var_o
  logical var_p

  var_m = 10

  allocate(var_l)
  allocate(var_l%var_j(var_m))
  allocate(var_l%var_h%var_f)
  allocate(var_l%var_k%var_c(var_m))
  allocate(var_l%var_k%var_b)
  allocate(var_l%var_h%var_g%var_c(var_m))
  allocate(var_l%var_h%var_g%var_b)

! Present checking and sending separately as if we send it in a single map/present check it in a single map
! we end up allocating bits and pieces implicitly due to the implicit binding parent map, but this seems
! analogous to Clang behaviour. It may need a bit of a rethink though, or a verification if it's the correct
! OpenMP behaviour.

!$omp target enter data map(ref_ptr, to: var_l%var_j)
!$omp target enter data map(ref_ptr, present, storage: var_l%var_j)

!$omp target enter data map(ref_ptr, to: var_l%var_k%var_c)
!$omp target enter data map(ref_ptr, present, storage: var_l%var_k%var_c)

!$omp target enter data map(ref_ptr, to: var_l%var_h%var_g%var_c)
!$omp target enter data map(ref_ptr, present, storage: var_l%var_h%var_g%var_c)

!$omp target enter data map(ref_ptr, to: var_l%var_h%var_g%var_b)
!$omp target enter data map(ref_ptr, present, storage: var_l%var_h%var_g%var_b)

!$omp target enter data map(ref_ptr, to: var_l%var_k%var_b)
!$omp target enter data map(ref_ptr, present, storage: var_l%var_k%var_b)

!$omp target enter data map(ref_ptr, to: var_l%var_h%var_f)
!$omp target enter data map(ref_ptr, present, storage: var_l%var_h%var_f)


! print *, "var_l%var_j ref_ptr on device"
print *, "var_l%var_k%var_c ref_ptr on device"
print *, "var_l%var_k%var_b ref_ptr on device"
print *, "var_l%var_h%var_f ref_ptr on device"
print *, "var_l%var_h%var_g%var_c ref_ptr on device"
print *, "var_l%var_h%var_g%var_b ref_ptr on device"


if(omp_target_is_present(C_LOC(var_l%var_j), omp_get_default_device()) == 0) then
   print *, "After enter: var_l%var_j ref_ptee NOT on device"
else
   print *, "After enter: var_l%var_j ref_ptee  on device"
   print*, "======= FORTRAN Test Failed! ======="
   stop 1
endif

if(omp_target_is_present(C_LOC(var_l%var_k%var_c), omp_get_default_device()) == 0) then
   print *, "After enter: var_l%var_k%var_c ref_ptee NOT on device"
else
   print *, "After enter: var_l%var_k%var_c ref_ptee on device"
   print*, "======= FORTRAN Test Failed! ======="
   stop 1
endif

if(omp_target_is_present(C_LOC(var_l%var_k%var_b), omp_get_default_device()) == 0) then
   print *, "After enter: var_l%var_k%var_b ref_ptee NOT on device"
else
   print *, "After enter: var_l%var_k%var_b ref_ptee on device"
   print*, "======= FORTRAN Test Failed! ======="
   stop 1
endif

if(omp_target_is_present(C_LOC(var_l%var_h%var_f), omp_get_default_device()) == 0) then
   print *, "After enter: var_l%var_h%var_f ref_ptee NOT on device"
else
   print *, "After enter: var_l%var_h%var_f ref_ptee on device"
   print*, "======= FORTRAN Test Failed! ======="
   stop 1
endif

if(omp_target_is_present(C_LOC(var_l%var_h%var_g%var_c), omp_get_default_device()) == 0) then
   print *, "After enter: var_l%var_h%var_g%var_c ref_ptee NOT on device"
else
   print *, "After enter: var_l%var_h%var_g%var_c ref_ptee on device"
   print*, "======= FORTRAN Test Failed! ======="
   stop 1
endif

if(omp_target_is_present(C_LOC(var_l%var_h%var_g%var_b), omp_get_default_device()) == 0) then
   print *, "After enter: var_l%var_h%var_g%var_b ref_ptee NOT on device"
else
   print *, "After enter: var_l%var_h%var_g%var_b ref_ptee on device"
   print*, "======= FORTRAN Test Failed! ======="
   stop 1
endif

! Slightly different from non-descriptor parent cases, the mapping should auotmatically
! map the parent descriptor and the data. This might not be needed in ref_ptr/ptee cases
! but it's needed for regular mappings, so for the moment it's left as it is until we run
! into a case where we need to re-evaluate it.
if(omp_target_is_present(C_LOC(var_l), omp_get_default_device()) == 0) then
   print *, "After enter: var_l parent structure NOT on device"
   print*, "======= FORTRAN Test Failed! ======="
   stop 1
else
   print *, "After enter: var_l parent structure on device"
endif

!$omp target enter data map(ref_ptee, to: var_l%var_j)
!$omp target enter data map(ref_ptee, to: var_l%var_k%var_c)
!$omp target enter data map(ref_ptee, to: var_l%var_k%var_b)
!$omp target enter data map(ref_ptee, to: var_l%var_h%var_f)
!$omp target enter data map(ref_ptee, to: var_l%var_h%var_g%var_c)
!$omp target enter data map(ref_ptee, to: var_l%var_h%var_g%var_b)

if(omp_target_is_present(C_LOC(var_l%var_j), omp_get_default_device()) == 0) then
   print *, "After enter: var_l%var_j ref_ptee NOT on device"
   print*, "======= FORTRAN Test Failed! ======="
   stop 1
else
   print *, "After enter: var_l%var_j ref_ptee  on device"
endif

if(omp_target_is_present(C_LOC(var_l%var_k%var_c), omp_get_default_device()) == 0) then
   print *, "After enter: var_l%var_k%var_c ref_ptee NOT on device"
   print*, "======= FORTRAN Test Failed! ======="
   stop 1
else
   print *, "After enter: var_l%var_k%var_c ref_ptee on device"
endif

if(omp_target_is_present(C_LOC(var_l%var_k%var_b), omp_get_default_device()) == 0) then
   print *, "After enter: var_l%var_k%var_b ref_ptee NOT on device"
   print*, "======= FORTRAN Test Failed! ======="
   stop 1
else
   print *, "After enter: var_l%var_k%var_b ref_ptee on device"
endif

if(omp_target_is_present(C_LOC(var_l%var_h%var_f), omp_get_default_device()) == 0) then
   print *, "After enter: var_l%var_h%var_f ref_ptee NOT on device"
   print*, "======= FORTRAN Test Failed! ======="
   stop 1
else
   print *, "After enter: var_l%var_h%var_f ref_ptee on device"
endif

if(omp_target_is_present(C_LOC(var_l%var_h%var_g%var_c), omp_get_default_device()) == 0) then
   print *, "After enter: var_l%var_h%var_g%var_c ref_ptee NOT on device"
   print*, "======= FORTRAN Test Failed! ======="
   stop 1
else
   print *, "After enter: var_l%var_h%var_g%var_c ref_ptee on device"
endif

if(omp_target_is_present(C_LOC(var_l%var_h%var_g%var_b), omp_get_default_device()) == 0) then
   print *, "After enter: var_l%var_h%var_g%var_b ref_ptee NOT on device"
   print*, "======= FORTRAN Test Failed! ======="
   stop 1
else
   print *, "After enter: var_l%var_h%var_g%var_b ref_ptee on device"
endif

   print *, "======= FORTRAN Test Passed! ======="
end program
