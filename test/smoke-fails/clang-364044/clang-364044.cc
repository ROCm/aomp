void a() {                                      
#pragma omp target teams distribute parallel for
  for (int b = 0; b < 0; b++)                   
    continue;                                   
}                                               
