int main() {
  int x = 0;
#pragma omp target data map(self : x)
  {
#pragma omp target map(x)
    {
// unspecified behaviour. It is not clear what x is used. Maybe a warning should
// be issued to inform the programmer.
#pragma omp atomic update
      x++;
    }
  }
}