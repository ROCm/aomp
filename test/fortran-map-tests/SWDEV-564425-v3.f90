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

  type(dtype_c), target :: var_l
  integer :: var_m, var_n, var_o
  logical var_p

  var_m = 10

  allocate(var_l%var_j(var_m))
  allocate(var_l%var_h%var_f)
  allocate(var_l%var_k%var_c(var_m))
  allocate(var_l%var_k%var_b)
  allocate(var_l%var_h%var_g%var_c(var_m))
  allocate(var_l%var_h%var_g%var_b)

!$omp target enter data map(ref_ptee, to: var_l%var_j, var_l%var_k%var_c, var_l%var_k%var_b, var_l%var_h%var_f, var_l%var_h%var_g%var_c, var_l%var_h%var_g%var_b)

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

!$omp target enter data map(ref_ptr, to: var_l%var_j, var_l%var_k%var_c, var_l%var_k%var_b, var_l%var_h%var_f, var_l%var_h%var_g%var_c, var_l%var_h%var_g%var_b)
!$omp target enter data map(ref_ptr, present, storage: var_l%var_j, var_l%var_k%var_c, var_l%var_k%var_b, var_l%var_h%var_f, var_l%var_h%var_g%var_c, var_l%var_h%var_g%var_b)

print *, "var_l%var_j ref_ptr on device"
print *, "var_l%var_k%var_c ref_ptr on device"
print *, "var_l%var_k%var_b ref_ptr on device"
print *, "var_l%var_h%var_f ref_ptr on device"
print *, "var_l%var_h%var_g%var_c ref_ptr on device"
print *, "var_l%var_h%var_g%var_b ref_ptr on device"

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

! Need to decide on the semantics of what should/should not happen to the parent
! when we do ref_ptr/ptee maps.
if (omp_target_is_present(C_LOC(var_l), omp_get_default_device()) == 0) then
   print *, "After enter: var_l parent structure NOT on device"
   print*, "======= FORTRAN Test Failed! ======="
   stop 1
else
   print *, "After enter: var_l parent structure on device"
endif

 !$omp target
   do var_n = 1, var_m
      var_l%var_j(var_n) = 20
      var_l%var_k%var_c(var_n) = 30
      var_l%var_h%var_g%var_c(var_n) = 40
   end do
   var_l%var_k%var_b = 5
   var_l%var_h%var_f = 20
   var_l%var_h%var_g%var_b = 25
 !$omp end target

!$omp target exit data map(ref_ptr_ptee, from: var_l%var_j, var_l%var_k%var_c, var_l%var_k%var_b, var_l%var_h%var_f, var_l%var_h%var_g%var_c, var_l%var_h%var_g%var_b)

   do var_n = 1, var_m
      if (var_l%var_j(var_n) /= 20) then
            print*, "======= FORTRAN Test Failed! ======="
            stop 1
      end if
   end do

   do var_n = 1, var_m
      if (var_l%var_k%var_c(var_n) /= 30) then
            print*, "======= FORTRAN Test Failed! ======="
            stop 1
      end if
   end do

   do var_n = 1, var_m
      if (var_l%var_h%var_g%var_c(var_n) /= 40) then
            print*, "======= FORTRAN Test Failed! ======="
            stop 1
      end if
   end do

   if (var_l%var_k%var_b /= 5) then
      print*, "======= FORTRAN Test Failed! ======="
      stop 1
   end if

   if (var_l%var_h%var_f /= 20) then
      print*, "======= FORTRAN Test Failed! ======="
      stop 1
   end if

   if (var_l%var_h%var_g%var_b /= 25) then
      print*, "======= FORTRAN Test Failed! ======="
      stop 1
   end if

   print *, var_l%var_j
   print *, var_l%var_k%var_c
   print *, var_l%var_h%var_g%var_c

   print *, var_l%var_k%var_b
   print *, var_l%var_h%var_f
   print *, var_l%var_h%var_g%var_b

   print *, "======= FORTRAN Test Passed! ======="
end program
