void foo(int n) {
  #pragma omp target device(n)
  ;
  #pragma omp target device(device_num: n)
  ;
}
int main() {
	 return 0;
}

