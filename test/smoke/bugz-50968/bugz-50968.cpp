#include <iostream>

template<int N>
struct A
{
	char arr[N];
	int b = 0;

	A()
	{
		arr[0] = 0;
	}
};

template<int N>
void f()
{
	int b = 0;

#pragma omp target map(tofrom: b)
	{
#pragma omp teams distribute
		for(int i = 0; i < 1; ++i)
		{
			A<N> a;
			b = a.arr[0]; // only this fails
			b = a.b += sizeof(a);
		}
	}

	std::cout << b << std::endl;
}

int main()
{
	//OK:
	f<58>();

	//[GPU Memory Error] Addr: 0x0 Reason: Page not present or supervisor privilege.:
	// https://bugs.llvm.org/show_bug.cgi?id=50968
	f<65>();

	//[GPU Memory Error] Addr: 0x0 Reason: Page not present or supervisor privilege.:
	// https://bugs.llvm.org/show_bug.cgi?id=50968
	f<30<<10>();
}
