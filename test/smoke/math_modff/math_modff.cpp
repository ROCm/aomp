#include <cmath>
using namespace std;

int main(int argc, char **argv)
{
  #pragma omp target
  {
    float intpart, res;
    res = modff(1.1f, &intpart);
  }

  #pragma omp target
  {
    double intpart, res;
    res = modf(1.1, &intpart);
  }
  return 0;
}
