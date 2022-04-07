#include <cmath>

template<typename T>
void test_modf(T x)
{
  T dx;
  int intx;

  #pragma omp target map(from: intx, dx)
  {
    T ipart;
    dx = std::modf(x, &ipart);
    intx = static_cast<int>(ipart);
  }
}

int main()
{

#if !defined(C_ONLY)
  test_modf<double>(1.0);
  test_modf<float>(1.0);
#endif

  #pragma omp target
  {
    double intpart, res;
    res = modf(1.1, &intpart);
  }

  #pragma omp target
  {
    float intpart, res;
    res = modff(1.1f, &intpart);
  }

}
