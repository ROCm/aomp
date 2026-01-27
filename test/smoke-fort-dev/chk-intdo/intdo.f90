!
!Copyright © Advanced Micro Devices, Inc., or its affiliates.
!
!SPDX-License-Identifier:  MIT
! 
! RUN: %flang %flags %openmp_flags -fopenmp-version=60 %s -o %t.exe
! RUN: %t.exe | $FILECHECK %s --match-full-lines

program interchange_intdo
  integer :: i, j
  print *, 'do'

  !$OMP INTERCHANGE
  do i = 7, 15, 3
    do j = -1, 1
      print '("i=", I0, " j=", I0)', i, j
    end do
  end do
  !$OMP END INTERCHANGE

  print *, 'done'
end program


! CHECK:      do
! CHECK-NEXT: i=7 j=-1
! CHECK-NEXT: i=10 j=-1
! CHECK-NEXT: i=13 j=-1
! CHECK-NEXT: i=7 j=0
! CHECK-NEXT: i=10 j=0
! CHECK-NEXT: i=13 j=0
! CHECK-NEXT: i=7 j=1
! CHECK-NEXT: i=10 j=1
! CHECK-NEXT: i=13 j=1
! CHECK-NEXT: done
