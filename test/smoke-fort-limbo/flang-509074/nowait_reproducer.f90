! Build Type:
!    - Debug: does not compile
!       - `inlinable function call in a function with debug info must have a !dbg location`
!    - Release: compiles

program nowait_reproducer
   use omp_lib
   implicit none

   integer, parameter :: n = 100
   real, allocatable :: array(:)
   integer :: i
   logical :: success

   ! Allocate array
   allocate (array(n))

   ! Initialize the array
   array = 0.0

   !$omp target map(tofrom: array) nowait
   !$omp parallel do
   do i = 1, n
      array(i) = real(i)
   end do
   !$omp end parallel do
   !$omp end target

   ! Check results
   success = .true.  ! Assume success initially
   do i = 1, n
      if (array(i) /= real(i)) then
         success = .false.
         print *, "Error at index ", i, ": ", array(i)
      end if
   end do

   ! Print final status
   if (success) then
      print *, "Success: All values are as expected"
   else
      print *, "Errors found in the computation"
   end if

   ! Deallocate array
   deallocate (array)
end program nowait_reproducer
