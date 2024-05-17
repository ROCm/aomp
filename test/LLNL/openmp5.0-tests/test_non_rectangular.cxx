#include <iostream>

int main(int argc, char **argv){

  const int N=100;
  double data[N][N];


for( int i=0;i<N;i++){
  for(int j=0;j<N;j++){
    data[i][j]=0.0;
  }
  data[i][i]=10*i;
 }


#pragma omp target map(tofrom: data)
#pragma omp parallel for collapse(2)
  for( int i=0;i<N;i++){
    //data[i][i]=10*i;
    for(int j=i;j<N;j++){
      data[i][j]+=i+j;
    }
  }
	

  int sum = 0.0;
  
  for( int i=0;i<N;i++){
    for(int j=i;j<N;j++){
      sum+= data[i][j]-(i+j);
    }
  }

  if (sum!=(N*(N-1)*5)){
    std::cout<<"Non rectangular loop test:: FAIL\n";
    std::cout<<"Sum is "<<sum<<"!="<<(N*(N-1)*5)<<"\n";
    return 1;
  }
  else {
    std::cout<<"Non rectangular loop test:: Pass\n";
    return 0;
  }
}
