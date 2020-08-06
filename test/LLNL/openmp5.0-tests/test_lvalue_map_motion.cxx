#include <iostream>


int& passthrough(int &in){
  return in;
}

int main(int argc, char **argv){

  const int N=300;
  const int test_val=123;
  int data[N];
  double *p = new double[1];
  p[0]=test_val;
  int test_data = test_val;


  for( int i=0;i<N;i++) data[i]=i;
  data[10]=test_val;
#pragma omp target data map(tofrom : *p, data[10],passthrough(test_data))  
#pragma omp target 
  {
    p[0]=test_val+1;
    data[10]=test_val+2;
    passthrough(test_data)=test_val+3;
  }
  
  if (p[0]==(test_val+1)) std::cout<<"lvalue map test 1 pass\n"; else std::cout<<"lvalue map test 1 FAIL \n" << p[0] <<" != "<<(test_val+1)<<"\n";
  if (data[10]==(test_val+2)) std::cout<<"lvalue map test 2 pass\n"; else std::cout<<"lvalue map test 2 FAIL \n" << data[10] <<" != "<<(test_val+2)<<"\n";
  if (passthrough(test_data)==(test_val+3)) std::cout<<"lvalue map test 3 pass\n"; else std::cout<<"lvalue map test 3 FAIL \n" << passthrough(test_data) <<" != "<<(test_val+3)<<"\n";


#pragma omp target enter data map(to : *p, data[10],passthrough(test_data)) 
  
  p[0] = test_val;
  data[10]=test_val;
  test_data = test_val;
  
#pragma omp target update to( *p, data[10],passthrough(test_data)) 
  
#pragma omp target 
  {
    p[0]=test_val+1;
    data[10]=test_val+2;
    passthrough(test_data)=test_val+3;
  }

#pragma omp target update from(*p, data[10],passthrough(test_data))
  
  if (p[0]==(test_val+1)) std::cout<<"lvalue map test 1 pass\n"; else std::cout<<"lvalue map test 1 FAIL \n" << p[0] <<" != "<<(test_val+1)<<"\n";
  if (data[10]==(test_val+2)) std::cout<<"lvalue map test 2 pass\n"; else std::cout<<"lvalue map test 2 FAIL \n" << data[10] <<" != "<<(test_val+2)<<"\n";
  if (passthrough(test_data)==(test_val+3)) std::cout<<"lvalue map test 3 pass\n"; else std::cout<<"lvalue map test 3 FAIL \n" << passthrough(test_data) <<" != "<<(test_val+3)<<"\n";
  
}
