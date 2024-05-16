! mimic NWChem tgt_sd_t_s1_1 kernel

#define simd schedule(static,1)

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

  n1 = 24
  n2 = 24
  n3 = 24
  n4 = 24
  n5 = 24
  n6 = 24

  nn = n1*n2*n3*n4*n5*n6

!$omp target teams distribute parallel do simd
  do ii = 1, nn
     t = ii-1
     q = t / n6
     r = t - q*n6
     i6 = l6 + r

     t = q
     q = t / n5
     r = t - q*n5
     i5 = l5 + r

     t = q
     q = t / n4
     r = t - q*n4
     i4 = l4 + r

     t = q
     q = t / n3
     r = t - q*n3
     i3 = l3 + r

     t = q
     q = t / n2
     r = t - q*n2
     i2 = l2 + r

     t = q
     q = t / n1
     r = t - q*n1
     i1 = l1 + r
     
     b(i6,i5,i4,i3,i2,i1) = a(i6,i5,i4,i3,i2,i1) + i3
enddo     
!$omp end target teams distribute parallel do
  
!  write(6,*) b(1,1,1,1,1,1)
!  write(6,*) a(1,1,1,1,1,1)
  
  write(6,*) ((b(k,j,1,1,1,1),j=1,4),k=1,4)
  return
end
