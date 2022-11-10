#include <iostream>
#include <complex>

bool failed = false;

template<typename TC, typename T>
void test_complex_copy()
{
  TC a_check,a;
  __real__(a_check) = T(0.2);
  __imag__(a_check) = T(1);

 #pragma omp target map(from: a) 
  a = a_check;
  if ((std::abs(__real__(a) - __real__(a_check)) > 1e-6) ||
      (std::abs(__imag__(a) - __imag__(a_check)) > 1e-6) ) 
  { std::cout 
	    << "\n FAIL copy WITHOUT map(to:) failed" << std::endl
            << "      check value  :(" <<  __real__(a) << " + " << __imag__(a) << "i)  &:" << &a<< std::endl
	    << "      correct value:(" 
	    <<  __real__(a_check) << " + " << __imag__(a_check) << "i)  &:" << &a_check
            << std::endl;
    failed = true;
  } else {
    printf("\n copy without map(to:) worked\n");
    std::cout
	    << "     check value  :(" <<  __real__(a) << " + " << __imag__(a) << "i)  &:" << &a<< std::endl
	    << "     correct value:(" <<  __real__(a_check) << " + " << __imag__(a_check) << "i)  &:" << &a_check
            << std::endl;
  }

 #pragma omp target map(from: a)  map(to:a_check)
    a = a_check;

  if ((std::abs(__real__(a) - __real__(a_check)) > 1e-6) ||
      (std::abs(__imag__(a) - __imag__(a_check)) > 1e-6) )
  { std::cout 
	    << "\n FAIL copy WITH map(to:) failed" << std::endl
            << "    check value:(" <<  __real__(a) << " + " << __imag__(a) << "i)  &:" << &a<< std::endl
	    << "    correct value:(" 
	    <<  __real__(a_check) << " + " << __imag__(a_check) << "i)  &:" << &a_check
            << std::endl;
    failed = true;
  } else {
    printf("\n copy WITH map(to:) worked\n");
    std::cout 
            << "   check value  :(" <<  __real__(a) << " + " << __imag__(a) << "i)  &:" << &a<< std::endl
	    << "   correct value:(" 
	    <<  __real__(a_check) << " + " << __imag__(a_check) << "i)  &:" << &a_check
            << std::endl;
  }
}

int main()
{
  std::cout << "Testing double _complex" << std::endl;
  test_complex_copy<double _Complex,double>();
  std::cout << "\nTesting float _complex" << std::endl;
  test_complex_copy<float _Complex,float>();

  return failed;
}
