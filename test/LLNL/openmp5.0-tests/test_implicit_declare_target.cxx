#include <iostream>

double foo(int i);
// hoo gets an imlicit decalre target to since it is called in a tarfet region
double hoo(int i){
return i;
}

int main(int argc, char **argv){

  const int N=300;
  double data[N];
  

#pragma omp target enter data map(to: data)
  

#pragma omp target teams distribute parallel for 
  for( int i=0;i<N;i++){
    data[i]=foo(i);
    data[i]+=hoo(i);
}


#pragma omp target exit data map(from: data)

  int sum = 0.0;
  
  for( int i=0;i<N;i++){
    sum+= data[i]-i*i-i;
  }

  
  if (sum!=(N*(N-1)/2)){
    std::cout<<"Declare target test:: FAIL\n";
    std::cout<<"Sum is "<<sum<<"!="<<(N*(N-1)/2)<<"\n";
    return 1;
  }
  else {
    std::cout<<"Declare target test:: Pass\n";
    return 0;
  }
}


