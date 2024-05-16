#include <iostream>

int main(int argc, char **argv){

  const int N=300;
  double data[N];

#pragma omp target enter data map(to: data)
  
#pragma omp target teams
#pragma omp loop // Does not Compile
  for( int i=0;i<N;i++){
  data[i]=i*i+i;
}


#pragma omp target exit data map(from: data)

  int sum = 0.0;
  
  for( int i=0;i<N;i++){
    sum+= data[i]-i*i;
  }

  
  if (sum!=(N*(N-1)/2)){
    std::cout<<"Loop test:: FAIL\n";
    std::cout<<"Sum is "<<sum<<"!="<<(N*(N-1)/2)<<"\n";
    return 1;
  }
  else {
    std::cout<<"Loop test:: Pass\n";
    return 0;
  }
}
