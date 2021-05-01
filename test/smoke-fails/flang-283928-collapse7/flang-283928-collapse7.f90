! test of collapse value >3

 

  implicit integer (a-z)
  l1 = 1; l2 = 1; l3 = 1; l4 = 1; l5 = 1; l6 = 1; l7 = 1
  u1 = 5; u2 = 5; u3 = 5; u4 = 5; u5 = 5; u6 = 5; u7 = 5
  call foo(l1,l2,l3,l4,l5,l6,l7, u1,u2,u3,u4,u5,u6,u7)
  end

 

subroutine foo(l1,l2,l3,l4,l5,l6,l7, u1,u2,u3,u4,u5,u6,u7)
  implicit integer (a-z)
  real a(10,10,10,10,10,10,10)
  real b(10,10,10,10,10,10,10)
  
  a=3.0
  b=0.0

 

!$omp target teams distribute parallel do collapse(7)
  do i1 = l1, u1
   do i2 = l2, u2
    do i3 = l3, u3
     do i4 = l4, u4
      do i5 = l5, u5
       do i6 = l6, u6
        do i7 = l7, u7
          b(i1,i2,i3,i4,i5,i6,i7) = a(i1,i2,i3,i4,i5,i6,i7) + i3
        end do
       end do
      end do
     end do
    end do
   end do
  end do
!$omp end target teams distribute parallel do
  write(6,*) ((b(k,j,1,1,1,1,1),j=1,10),k=1,10)
  return
end
