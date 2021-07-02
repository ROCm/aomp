#include <iostream>

template<class T>
void f()
{
        int b = 0;

#pragma omp target map(tofrom: b)
        {
#pragma omp teams distribute
                for(int i = 0; i < 1; ++i)
                {
                        T a = 0;
#pragma omp parallel num_threads(64)
                        {
#pragma omp atomic update
                                a += 1;
                        }
                        b = static_cast<int>(a);
                }
        }

        std::cout << b << std::endl;
}

int main()
{
        // OK:
        f<int>();
        f<unsigned int>();

        // ICE:
        f<char>();
        f<short>();
        f<long long>();
        f<float>();
        f<double>();
}
