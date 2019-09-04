template<typename T>
class foo
{
public:
  foo()
  {
    #pragma omp target
    {
      T a;
    }
  }
};
