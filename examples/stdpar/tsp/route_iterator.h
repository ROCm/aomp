#ifndef ROUTE_ITERATOR_H_
#define ROUTE_ITERATOR_H_

#include <cstdint>
#include <cstdio>
#include <inttypes.h>
#include <cassert>

// =====================================================
// =====================================================
//! function to find factorial of given number
uint64_t factorial(uint64_t n)
{
    if (n == 0)
      return 1;
    return n * factorial(n - 1);
}

// =====================================================
// =====================================================
//! integer division, compute quotient and reminder
inline void int_div(uint64_t k, uint64_t divisor,
                    uint64_t& q, uint64_t& r)
{
  q = k / divisor;
  r = k - divisor*q;
}

// =====================================================
// =====================================================
//! swap values
template <typename T>
void my_swap(T& a, T& b)
{
    T c(a); a=b; b=c;
}

// =====================================================
// =====================================================
/**
 * Generate a unique permutation of N elements.
 */
template <int N>
class route_iterator
{
public:
  //! constructor
  route_iterator(uint64_t route_id);

  //! at the end of the route ?
  bool done() const { return m_i==(N-1); };

  //! first city of the route
  int first() {  m_i=0; return m_state[0]; };

  //! next city of the route
  int next() { m_i++; return m_state[m_i]; };

  void print();

private:
  uint64_t m_route_id;
  int m_i;
  int m_state[N];

}; // class route_iterator

// =====================================================
// =====================================================
template<int N>
route_iterator<N>::route_iterator(uint64_t route_id)
  : m_route_id(route_id),
    m_i(0)
{
  assert(m_route_id < factorial(N));

  // invalid value
  for (int k=0; k<N; ++k)
    m_state[k] = -1;

  // identity
  for (int k=0; k<N; ++k)
    m_state[k] = k;


  assert(N>=2);

  int divisor = 1;
  uint64_t iter = m_route_id;
  for(int k=1; k<N; ++k)
  {

    uint64_t q, r;
    int_div(iter/divisor, k+1, q, r);

    if (r > 0)
    {
      my_swap(m_state[k], m_state[k-r]);
    }

    divisor *= (k+1);
  }
}

// =====================================================
// =====================================================
template<int N>
void route_iterator<N>::print()
{
  printf("Permutation #%06" PRId64 " || ",m_route_id);

  // for (int k=0; k<N; ++k)
  //   printf("[%d] -> %d\n",k,m_state[k]);

  for (int k=0; k<N; ++k)
  {
    printf("%2d ",m_state[k]);
  }
  printf("\n");

} // route_iterator::print

#endif // ROUTE_ITERATOR_H_
