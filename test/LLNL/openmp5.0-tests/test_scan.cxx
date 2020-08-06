#include <iostream>

int main(int argc, char **argv){

  const int N=300;
  int data[N];



  for( int i=0;i<N;i++) data[i]=0;
  int scan_var = 0;
#pragma omp target data map(tofrom : data)  
#pragma omp target teams distribute parallel for reduction(inscan,+: scan_var)
  for( int i=0;i<N;i++){
    scan_var+=i+1;
#pragma omp scan inclusive(scan_var)
    data[i]+=scan_var;
  }

  //for( int i=0;i<N;i++) std::cout<<i<<" "<<data[i]<<"\n";


int err_count=0;
for(int i=0;i<N;i++)
  if (data[i]!=(i+1)*(i+2)/2) {
    std::cout<<"ERROR "<<data[i]<<" != "<<(i+1)*(i+2)/2<<"\n";
    err_count++;
  }
  
if (err_count){
  std::cout<<"Inclusive scan test:: FAIL\n";
  std::cout<<"Err count is "<<err_count<<"\n";
    return 1;
  }
  else {
    std::cout<<"Inclusive scan test:: Pass\n";
    return 0;
  }
}
