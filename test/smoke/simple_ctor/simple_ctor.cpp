#pragma omp declare target

// tries to cast without taking addrspace into consideration, crashes
__attribute__((used))
struct t
{
  t() {}
} o;

#pragma omp end declare target

int main() {}
