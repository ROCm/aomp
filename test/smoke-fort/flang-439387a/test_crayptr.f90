program test_crayptr
     implicit none
     integer*8 var(2,*) !works with gfortran, but not accepted by flang-new
     pointer(ivar,var)
     integer*8, allocatable :: location(:,:)
     integer i,j
 
     allocate(location(2,5))
     ivar = loc(location)
     do j=1,2
     do i=1,5
       var(j,i) = i
     enddo
     enddo
     write(*,*) location
 
end program test_crayptr
