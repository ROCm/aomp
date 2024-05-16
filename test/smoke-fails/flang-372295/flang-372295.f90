Program test_openmp_vector_in_routine

  implicit none

  interface

     Subroutine myroutine(a,b,j)

       implicit none

       !$omp declare target

       
       !real, allocatable, intent(inout) :: a(:,:) , b(:,:)
       real, intent(inout) :: a(:,:),b(:,:)
       integer, intent(in) :: j
            
     End Subroutine myroutine
     
  end interface

  
  integer i,j
  real, allocatable :: a(:,:)
  real, allocatable :: b(:,:)

  allocate( a(2000,100) )
  allocate( b(2000,100) )

  do j=1,100
     do i=1,2000 
        a(i,j) = -1
        b(i,j) = i + j * 2000
     enddo
  enddo


  !$omp target data map(tofrom:a,b)  
  
  !$omp target teams distribute  
  do j=1,100 
     call myroutine(a,b,j)
     !!a(10,j) = b(10,j)
  enddo
  !$omp end target teams distribute
  
  !$omp end target data
  
  print *, a(10,10), b(10,10)

  deallocate(a)
  deallocate(b)
  
End Program test_openmp_vector_in_routine



Subroutine myroutine(a,b,j)

    implicit none

!$omp declare target
  
  !real, allocatable, intent(inout) :: a(:,:) , b(:,:)
  real, intent(inout) :: a(:,:), b(:,:) 
  integer, intent(in) :: j

  !!local
  integer :: i

  
  do i=1,2000
     a(i,j) = b(i,j)
  enddo
     
End Subroutine myroutine



!//teams
!do 
!  //para
! do 
! // para 
! do 

 

 
