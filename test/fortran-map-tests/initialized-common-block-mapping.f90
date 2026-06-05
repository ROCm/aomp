! Copyright © Advanced Micro Devices, Inc., or its affiliates.
!
! SPDX-License-Identifier:  MIT

! Common blocks can have a special "edge-case" where they are initialized
! common blocks, which basically just means we give them some values on
! initialize. This, however, means the LLVM-IR for a common block is actually
! generated differently, the common block goes from a byte array into a typed
! tuple, so we need to be sure this still lowers and handles fine.

block data init_blocks
  implicit none

  integer :: var_a
  real :: var_b(5)
  common /cblock_a/ var_a, var_b

  integer :: var_c
  real :: var_d(5)
  common /cblock_b/ var_c, var_d

  data var_a / 100 /
  data var_b / 1.0, 2.0, 3.0, 4.0, 5.0 /

  data var_c / 200 /
  data var_d / 10.0, 20.0, 30.0, 40.0, 50.0 /

end block data

program prog_a
  implicit none

  integer :: var_a
  real :: var_b(5)
  common /cblock_a/ var_a, var_b
  !$omp declare target link(/cblock_a/)

  integer :: var_c
  real :: var_d(5)
  common /cblock_b/ var_c, var_d

  integer :: i

  print *, "=== Initial values (before target region) ==="
  print *, ""
  print *, "cblock_a (with declare target link):"
  print *, "  var_a =", var_a
  print *, "  var_b =", var_b
  print *, ""
  print *, "cblock_b (without declare target):"
  print *, "  var_c =", var_c
  print *, "  var_d =", var_d
  print *, ""

  !$omp target map(tofrom: /cblock_a/, /cblock_b/)
    var_a = var_a + 1000
    do i = 1, 5
      var_b(i) = var_b(i) * 2.0
    end do

    var_c = var_c + 2000
    do i = 1, 5
      var_d(i) = var_d(i) * 3.0
    end do
  !$omp end target

  print *, "=== Final values (after target region) ==="
  print *, ""
  print *, "cblock_a (with declare target link):"
  print *, "  var_a =", var_a
  print *, "  var_b =", var_b
  print *, ""
  print *, "cblock_b (without declare target):"
  print *, "  var_c =", var_c
  print *, "  var_d =", var_d
  print *, ""

  print *, "=== Validating results ==="

  if (var_a /= 1100) then
    print *, "FAILED: var_a expected 1100, got", var_a
    print *, "======= FORTRAN Test Failed! ======="
    stop 1
  end if

  if (var_b(1) /= 2.0 .or. var_b(2) /= 4.0 .or. &
      var_b(3) /= 6.0 .or. var_b(4) /= 8.0 .or. &
      var_b(5) /= 10.0) then
    print *, "FAILED: var_b expected [2.0, 4.0, 6.0, 8.0, 10.0], got", var_b
    print *, "======= FORTRAN Test Failed! ======="
    stop 1
  end if

  if (var_c /= 2200) then
    print *, "FAILED: var_c expected 2200, got", var_c
    print *, "======= FORTRAN Test Failed! ======="
    stop 1
  end if

  if (var_d(1) /= 30.0 .or. var_d(2) /= 60.0 .or. &
      var_d(3) /= 90.0 .or. var_d(4) /= 120.0 .or. &
      var_d(5) /= 150.0) then
    print *, "FAILED: var_d expected [30.0, 60.0, 90.0, 120.0, 150.0], got", var_d
    print *, "======= FORTRAN Test Failed! ======="
    stop 1
  end if

  print *, "======= FORTRAN Test Passed! ======="

end program prog_a
