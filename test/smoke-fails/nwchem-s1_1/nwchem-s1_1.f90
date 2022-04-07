! mimic NWChem tgt_sd_t_s1_1 kernel
! RL: do not redefine simd clause to be schedule(static, 1)
! RL: make the schedule clause usage be explicit

  implicit integer (a-z)
  l1 = 1; l2 = 1; l3 = 1; l4 = 1; l5 = 1; l6 = 1;
  u1 = 24; u2 = 24; u3 = 24; u4 = 24; u5 = 24; u6 = 24;
  call tgt_sd_t_s1_1(l1,l2,l3,l4,l5,l6, u1,u2,u3,u4,u5,u6)
  end

 

subroutine tgt_sd_t_s1_1(l1,l2,l3,l4,l5,l6, u1,u2,u3,u4,u5,u6)
  implicit integer (a-z)
  real a(24,24,24,24,24,24)
  real b(24,24,24,24,24,24)
  
  a=3.0
  b=0.0

 

!$omp target teams distribute parallel do schedule(static,1) collapse(6)
  do i1 = l1, u1
   do i2 = l2, u2
    do i3 = l3, u3
     do i4 = l4, u4
      do i5 = l5, u5
       do i6 = l6, u6
          b(i6,i5,i4,i3,i2,i1) = a(i6,i5,i4,i3,i2,i1) + i3
       end do
      end do
     end do
    end do
   end do
  end do
!$omp end target teams distribute parallel do

!  write(6,*) b(1,1,1,1,1,1)
!  write(6,*) a(1,1,1,1,1,1)
  
  write(6,*) ((b(k,j,1,1,1,1),j=1,4),k=1,4)
  return
end
