#include <stdio.h>
#include <math.h>

#define FUNC(x) sqrt(x)
#define TYPE float
#define EPS 1e-5

void vmul(TYPE*a, TYPE*b, TYPE*c, int N){
#pragma omp target map(to: a[0:N],b[0:N]) map(from:c[0:N])
#pragma omp teams distribute parallel for
   for(int i=0; i<N; i++) {
      c[i]=FUNC(a[i])*FUNC(b[i]);
   }
}

int main(){
    const int N = 100;
    TYPE a[N], b[N], c[N], validate[N];
    TYPE delta, delta_max=0;
    int flag=-1, flag_max=-1; // Mark Success
    for(int i=0; i<N; i++) {
        a[i]=i+1;
        b[i]=i+2;
        validate[i]=FUNC(a[i])*FUNC(b[i]);
    }

    vmul(a,b,c,N);

    for(int i=0; i<N; i++) {
        delta=fabs(c[i]-validate[i]);
        if(delta > EPS) {
//          print 1st bad index
            if(flag == -1)
              printf("First fail: c[%d](%g) != validate[%d](%g)  <%g>\n",
                i,c[i],i,validate[i],c[i]-validate[i]);
            if(delta > delta_max) {
              delta_max=delta;
              flag_max=flag;
            }
            flag = i;
        }
    }
    if(flag == -1){
        printf("Success\n");
        return 0;
    } else {
        printf("Last  fail: c[%d](%g) != validate[%d](%g)  <%g>\n",
          flag,c[flag],flag,validate[flag],c[flag]-validate[flag]);
        printf("Max   fail: c[%d](%g) != validate[%d](%g)  <%g>\n", flag_max,
          c[flag_max],flag_max,validate[flag_max],c[flag_max]-validate[flag_max]);
        printf("Fail\n");
        return 1;
    }
}

