program omp_taskwait_depend

use omp_lib

implicit none
!integer :: n = 1000000000
! smaller size for regression testing
integer :: n = 100000
double precision, allocatable, dimension(:)  :: a1, b1
double precision, allocatable, dimension(:)  :: a2, b2
double precision :: t0, t1, t2, t3, t4, ans

allocate(a1(n))
allocate(b1(n))
allocate(a2(n))
allocate(b2(n))

b1 = 2.0d0
b2 = 2.0d0
ans = b1(1) * b1(1)

!$omp target enter data map(to:b1, b2) map(alloc:a1, a2)
t0 = omp_get_wtime()

t1 = omp_get_wtime() - t0
call square(a1, b1, n)

t2 = omp_get_wtime() - t0
call square(a2, b2, n)

!$omp taskwait depend(in:a1)
t3 = omp_get_wtime() - t0
print *, "Task 1 start time:", 1e3 * t1, "elapsed time until now:", 1e3 * t3, "ms"

!$omp taskwait depend(in:a2)
t4 = omp_get_wtime() - t0
print *, "Task 2 start time:", 1e3 * t2, "elapsed time until now:", 1e3 * t4, "ms"

!$omp target update from(a1, a2)
print *, "a1(n) = ", a1(n), "expected = ", ans
print *, "a2(n) = ", a2(n), "expected = ", ans

!$omp target exit data map(delete:a1, a2, b1, b2)
deallocate(a1)
deallocate(a2)
deallocate(b1)
deallocate(b2)

contains
subroutine square(a, b, n)
implicit none
double precision, dimension(:), intent(out) :: a
double precision, dimension(:), intent(in) :: b
integer :: i, n
!$omp target teams distribute parallel do nowait depend(out:a)
do i=1,n
a(i) = b(i) * b(i)
end do

end subroutine
end program
