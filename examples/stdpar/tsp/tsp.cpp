#include "route_iterator.h"
#include "route_cost.h"
#include "tsp_utils.h"

// #include <thrust/iterator/counting_iterator.h>
// #include <thrust/system/omp/execution_policy.h>
// #include <thrust/system/omp/vector.h>
// #include <thrust/generate.h>
// #include <thrust/reduce.h>

#include <cstdio> // for printf
#include <cstdlib> // for EXIT_SUCCESS
#include <iostream>

#include <algorithm>
#include <chrono>
#include <execution>
#include <numeric>
#include <ranges>

using namespace std;
using namespace std::chrono;

// ============================================
// ============================================
void print_factorials(int N)
{
  for (int k=0; k<N; ++k)
    printf("factorial(%d)=%" PRId64 "\n",k,factorial(k));
}

// ============================================
// ============================================
template<int N>
void test_permutation()
{

  for (int k=0; k<factorial(N); ++k)
  {

    // create permutation
    route_iterator<N> rit(k);

    // print on screen for cross-check
    rit.print();

  }

} // test_permutation

// ============================================
// ============================================
void test_city_distance()
{
  auto cityIndex = makeCityMap();
  auto size = cityIndex.size();

  auto distances = init_distance_matrix();

  for (size_t from=0; from<size; ++from)
  {
    for (size_t to=0; to<size; ++to)
    {
      assert(distances[to+size*from] == distances[from+size*to]);
      printf("%05d ",distances[to+size*from]);
    }
    printf("\n");
  }

  int small_n = 5;
  auto distances_small = init_distance_matrix_small(small_n);

  for (int from=0; from<small_n; ++from)
  {
    for (int to=0; to<small_n; ++to)
    {
      assert(distances_small[to+small_n*from] == distances_small[from+small_n*to]);
      printf("%05d ",distances_small[to+small_n*from]);
    }
    printf("\n");
  }

} // test_city_distance


template<int N>
route_cost find_best_route(int const* distances)
{
  int X = factorial(N);
  return
    std::transform_reduce(std::execution::par_unseq,
		    	  std::views::iota(0, X).begin(),
		    	  std::views::iota(0, X).end(),
                          route_cost(),
                          [](route_cost x, route_cost y) { return x.cost < y.cost ? x : y; },
                          [=](int64_t i)
  {
    int cost = 0;
    route_iterator<N> it(i);

    // first city visited
    int from = it.first();

    // visited all other cities in the chosen route
    // and compute cost
    while (!it.done())
    {
      int to = it.next();
      cost += distances[to + N * from];
      from = to;
    }
    // update best_route -> reduction
    return route_cost(i, cost);
  });

}

// ============================================================
// ============================================================
//! \param[in] nbRepeat number of repeat (for accurate time measurement)

template <int N>
void solve_traveling_salesman(int nbRepeat = 1)
{
  auto distances_small = init_distance_matrix_small(N);
  route_cost best_route;

  auto start = high_resolution_clock::now();
  for (int i = 0; i<nbRepeat; ++i)
    best_route = find_best_route<N>(distances_small.data());

  auto end = high_resolution_clock::now();

  duration<double, milli> elapsed = (end - start) / nbRepeat;
  // print best route
  printf("Trav Salesman Prob N=%d, best route cost is: %d, average time is %f seconds\n",N, best_route.cost, elapsed.count());

  printf("Solution route is ");
  route_iterator<N> rit(best_route.route);
  rit.print();

} // solve_traveling_salesman

// ============================================================
// ============================================================
int main(int argc, char* argv[])
{

  // print factorials up to 14!
  // print_factorials(14);

  // dump permutations on screen
  // constexpr int N=4;
  // test_permutation<N>();

  // auto cityIndex = makeCityMap();
  // for (auto city : cityIndex)
  //   std::cout << city.first << " " << city.second << "\n";

  //test_city_distance();

  int n = 10;
  if (argc>1)
    n = atoi(argv[1]);

  if (n==10)
    solve_traveling_salesman<10>();
  else if (n==11)
    solve_traveling_salesman<11>();
  else if (n==12)
    solve_traveling_salesman<12>();
  else if (n==13)
    solve_traveling_salesman<13>();
  else if (n==14)
    solve_traveling_salesman<14>();

  return EXIT_SUCCESS;

} // main
