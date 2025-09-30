
! RUN: %flang %flags %openmp_flags -fopenmp-version=60 %s -o %t.exe
! RUN: %t.exe | FileCheck %s --match-full-lines

program interchange_intdo
  integer :: i, j
  print *, 'do'

  !$OMP INTERCHANGE PERMUTATION(2,3,1)
  do i = 7, 15, 3
    do j = -1, 1, 2
      do k = 3, 1, -1
        print '("i=", I0, " j=", I0, " k=", I0)', i, j, k
      end do
    end do
  end do
  !$OMP END INTERCHANGE

  print *, 'done'
end program


! CHECK:      do
! CHECK-NEXT: i=7 j=-1 k=3
! CHECK-NEXT: i=10 j=-1 k=3
! CHECK-NEXT: i=13 j=-1 k=3
! CHECK-NEXT: i=7 j=-1 k=2
! CHECK-NEXT: i=10 j=-1 k=2
! CHECK-NEXT: i=13 j=-1 k=2
! CHECK-NEXT: i=7 j=-1 k=1
! CHECK-NEXT: i=10 j=-1 k=1
! CHECK-NEXT: i=13 j=-1 k=1
! CHECK-NEXT: i=7 j=1 k=3
! CHECK-NEXT: i=10 j=1 k=3
! CHECK-NEXT: i=13 j=1 k=3
! CHECK-NEXT: i=7 j=1 k=2
! CHECK-NEXT: i=10 j=1 k=2
! CHECK-NEXT: i=13 j=1 k=2
! CHECK-NEXT: i=7 j=1 k=1
! CHECK-NEXT: i=10 j=1 k=1
! CHECK-NEXT: i=13 j=1 k=1
! CHECK-NEXT: done
