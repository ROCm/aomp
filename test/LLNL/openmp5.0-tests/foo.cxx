// goo gets an imolicit declare target since it is called in foo
double goo(int i){
  return (double)(i*i+i);
}

double foo(int i){
  return goo(i);
}

#pragma omp declare target to(foo) // device_type(nohost)
