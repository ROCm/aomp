program main
    implicit none

     !$omp requires unified_shared_memory

   !only fails if a is in a type
    TYPE :: polytype
      REAL, DIMENSION(:),ALLOCATABLE :: a
    END TYPE
    TYPE(polytype) :: poly

    integer,parameter :: n = 10
    integer :: j

         ALLOCATE(poly%a(1:3))
      !$omp target teams distribute parallel do private(poly)
      do j=1,n
        poly%a = 2.0_8 !array assign in kernel to a type member is the issue
      enddo
      !$omp end target teams distribute parallel do

         DEALLOCATE(poly%a)
end program
