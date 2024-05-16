#include <iostream>
#include <iomanip>

void bc(int *a, int N, int value){
  for(int i=0;i<=N+1;i+=N+1) for(int j=0;j<(N+2);j++) a[i+(N+2)*j]=value;
  for(int j=0;j<=N+1;j+=N+1) for(int i=0;i<(N+2);i++) a[i+(N+2)*j]=value;
}


void bc_offload(int *a, int N, int value){
#pragma omp target teams distribute parallel for 
  for(int i=0;i<=N+1;i+=N+1) for(int j=0;j<(N+2);j++) a[i+(N+2)*j]=value;
#pragma omp target teams distribute parallel for 
  for(int j=0;j<=N+1;j+=N+1) for(int i=0;i<(N+2);i++) a[i+(N+2)*j]=value;
}


  int check(int *a, int N, int value){
  int err=0;
  for(int i=0;i<=N+1;i+=N+1) for(int j=0;j<(N+2);j++) if (a[i+(N+2)*j]!=value) err++;
  for(int j=0;j<=N+1;j+=N+1) for(int i=0;i<(N+2);i++) if (a[i+(N+2)*j]!=value) err++;
  return err;
  }
void printarray(int *data, int N){
  for (int i=0;i<=N+1;i++) {
    for (int j=0;j<=N+1;j++) std::cout<<std::setw(2) << std::setfill('0') <<data[i+(N+2)*j]<<" ";
    std::cout<<"\n";
  }
}

int main(int argc, char **argv){

  const int N=9;
  int *data;
  data = new int[(N+2)*(N+2)];

  for (int i=1;i<=N;i++) for (int j=1;j<=N;j++) data[i+(N+2)*j]=i*10+j;
  
  printarray(data,N);

#pragma omp target enter data map(to:data[0:(N+2)*(N+2)])
  
  bc(data,N,-1);

  printarray(data,N);

#pragma omp target update to( (([N+2][N+2])data)[0:N+2][0], (([N+2][N+2])data)[0:N+2][N+1])
#pragma omp target update to( (([N+2][N+2])data)[0][0:N+2], (([N+2][N+2])data)[0][N+1][0:N+2])

  bc_offload(data,N,-2);
  
#pragma omp target update from( (([N+2][N+2])data)[0:N+2][0], (([N+2][N+2])data)[0:N+2][N+1])
#pragma omp target update from( (([N+2][N+2])data)[0][0:N+2], (([N+2][N+2])data)[0][N+1][0:N+2])

  
    //#pragma omp target exit data map(from:data[0:(N+2)*(N+2)])

    if (check(data,N,-2)) {
      std::cout<<"\n\n\nContiguous update test :: FAIL\nBoundary values should be -2\n\n\n";
      printarray(data,N);
      return 1;
      
    } else {
      std::cout<<"Contiguous update test :: PASS\n";
      return 0;
    }
 

  

}
