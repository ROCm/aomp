#include <string>

#pragma omp declare target

void *foo(std::string use_case, int num, int val, int *p)
{

  if (use_case=="DEV")
  {
    for( int i = 0; i < num; ++i )
    {
      p[i] = (i<num/2)?val:0;
    }

    return p;
  }

  return p;
}
#pragma omp end declare target

int main()
{
  std::string use_case="DEV";
  int num=16;
  int *res= (int *) malloc(num * sizeof(int));

  foo(use_case, num,4,res);

  return 0;
}
