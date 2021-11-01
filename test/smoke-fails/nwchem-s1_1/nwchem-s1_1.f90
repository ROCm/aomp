! mimic NWChem tgt_sd_t_s1_1 kernel

#define simd schedule(static,1)

  implicit integer (a-z)
  l1 = 1; l2 = 1; l3 = 1; l4 = 1; l5 = 1; l6 = 1;
  u1 = 16; u2 = 16; u3 = 16; u4 = 16; u5 = 16; u6 = 16;
  call tgt_sd_t_s1_1(l1,l2,l3,l4,l5,l6, u1,u2,u3,u4,u5,u6)
  end

 

subroutine tgt_sd_t_s1_1(l1,l2,l3,l4,l5,l6, u1,u2,u3,u4,u5,u6)
  implicit integer (a-z)
  real a(16,16,16,16,16,16)
  real b(16,16,16,16,16,16)
  
  a=3.0
  b=0.0

 

!$omp target teams distribute parallel do simd collapse(6) num_teams(240)
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
  
  write(6,*) ((b(k,j,1,1,1,1),j=1,16),k=1,16)
  return
end
