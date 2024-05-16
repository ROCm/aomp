void vsum_302(int*a, int*b, int*c, int N){
#pragma omp target teams map(to: a[0:N],b[0:N]) map(from:c[0:N])
#pragma omp distribute parallel for
   for(int i=0;i<N;i++) {
      c[i]=a[i]+b[i];
   }
 }
