#ifndef TSP_H_
#define TSP_H_

#include <map>
#include <vector>
#include <string>

// ==================================================
// ==================================================
auto makeCityMap() -> std::map<std::string, int>
{
  std::map<std::string, int> city2index;

  city2index.insert(std::make_pair("Paris",0));
  city2index.insert(std::make_pair("Marseille",1));
  city2index.insert(std::make_pair("Bordeaux",2));
  city2index.insert(std::make_pair("Toulouse",3));
  city2index.insert(std::make_pair("Brest",4));
  city2index.insert(std::make_pair("Nantes",5));
  city2index.insert(std::make_pair("Lille",6));
  city2index.insert(std::make_pair("Metz",7));
  city2index.insert(std::make_pair("Nancy",8));
  city2index.insert(std::make_pair("Lyon",9));
  city2index.insert(std::make_pair("Clermont-Ferrand",10));
  city2index.insert(std::make_pair("Strasbourg",11));
  city2index.insert(std::make_pair("Tours",12));
  city2index.insert(std::make_pair("Limoges",13));

  return city2index;
}

// ==================================================
// ==================================================
std::vector<int> init_distance_matrix()
{

  // distances à vol d'oiseau

  auto cityIndex = makeCityMap();
  auto size = cityIndex.size();

  std::vector<int> distances(size*size);


  // ========================================================
  auto from = cityIndex["Paris"];
  auto to   = cityIndex["Paris"];
  distances[to + size*from] = 0;
  // ========================================================

  to   = cityIndex["Marseille"];
  distances[to + size*from] = 661;

  to    = cityIndex["Bordeaux"];
  distances[to + size*from] = 500;

  to    = cityIndex["Toulouse"];
  distances[to + size*from] = 589;

  to    = cityIndex["Brest"];
  distances[to + size*from] = 506;

  to    = cityIndex["Nantes"];
  distances[to + size*from] = 343;

  to    = cityIndex["Lille"];
  distances[to + size*from] = 204;

  to    = cityIndex["Metz"];
  distances[to + size*from] = 281;

  to    = cityIndex["Nancy"];
  distances[to + size*from] = 282;

  to    = cityIndex["Lyon"];
  distances[to + size*from] = 392;

  to    = cityIndex["Clermont-Ferrand"];
  distances[to + size*from] = 347;

  to    = cityIndex["Strasbourg"];
  distances[to + size*from] = 397;

  to    = cityIndex["Tours"];
  distances[to + size*from] = 205;

  to    = cityIndex["Limoges"];
  distances[to + size*from] = 347;

  // ========================================================
  from = cityIndex["Marseille"];
  to   = cityIndex["Marseille"];
  distances[to + size*from] = 0;
  // ========================================================

  to    = cityIndex["Paris"];
  distances[to + size*from] = 661;

  to    = cityIndex["Bordeaux"];
  distances[to + size*from] = 506;

  to    = cityIndex["Toulouse"];
  distances[to + size*from] = 319;

  to    = cityIndex["Brest"];
  distances[to + size*from] = 951;

  to    = cityIndex["Nantes"];
  distances[to + size*from] = 696;

  to    = cityIndex["Lille"];
  distances[to + size*from] = 835;

  to    = cityIndex["Metz"];
  distances[to + size*from] = 651;

  to    = cityIndex["Nancy"];
  distances[to + size*from] = 604;

  to    = cityIndex["Lyon"];
  distances[to + size*from] = 278;

  to    = cityIndex["Clermont-Ferrand"];
  distances[to + size*from] = 330;

  to    = cityIndex["Strasbourg"];
  distances[to + size*from] = 617;

  to    = cityIndex["Tours"];
  distances[to + size*from] = 585;

  to    = cityIndex["Limoges"];
  distances[to + size*from] = 431;

  // ========================================================
  from = cityIndex["Bordeaux"];
  to   = cityIndex["Bordeaux"];
  distances[to + size*from] = 0;
  // ========================================================

  to    = cityIndex["Paris"];
  distances[to + size*from] = 500;

  to    = cityIndex["Marseille"];
  distances[to + size*from] = 506;

  to    = cityIndex["Toulouse"];
  distances[to + size*from] = 212;

  to    = cityIndex["Brest"];
  distances[to + size*from] = 496;

  to    = cityIndex["Nantes"];
  distances[to + size*from] = 276;

  to    = cityIndex["Lille"];
  distances[to + size*from] = 700;

  to    = cityIndex["Metz"];
  distances[to + size*from] = 700;

  to    = cityIndex["Nancy"];
  distances[to + size*from] = 671;

  to    = cityIndex["Lyon"];
  distances[to + size*from] = 436;

  to    = cityIndex["Clermont-Ferrand"];
  distances[to + size*from] = 306;

  to    = cityIndex["Strasbourg"];
  distances[to + size*from] = 760;

  to    = cityIndex["Tours"];
  distances[to + size*from] = 301;

  to    = cityIndex["Limoges"];
  distances[to + size*from] = 182;

  // ========================================================
  from = cityIndex["Toulouse"];
  to   = cityIndex["Toulouse"];
  distances[to + size*from] = 0;
  // ========================================================

  to    = cityIndex["Paris"];
  distances[to + size*from] = 589;

  to    = cityIndex["Marseille"];
  distances[to + size*from] = 319;

  to    = cityIndex["Bordeaux"];
  distances[to + size*from] = 212;

  to    = cityIndex["Brest"];
  distances[to + size*from] = 703;

  to    = cityIndex["Nantes"];
  distances[to + size*from] = 466;

  to    = cityIndex["Lille"];
  distances[to + size*from] = 792;

  to    = cityIndex["Metz"];
  distances[to + size*from] = 713;

  to    = cityIndex["Nancy"];
  distances[to + size*from] = 674;

  to    = cityIndex["Lyon"];
  distances[to + size*from] = 360;

  to    = cityIndex["Clermont-Ferrand"];
  distances[to + size*from] = 275;

  to    = cityIndex["Strasbourg"];
  distances[to + size*from] = 737;

  to    = cityIndex["Tours"];
  distances[to + size*from] = 426;

  to    = cityIndex["Limoges"];
  distances[to + size*from] = 249;

  // ========================================================
  from = cityIndex["Brest"];
  to   = cityIndex["Brest"];
  distances[to + size*from] = 0;
  // ========================================================

  to    = cityIndex["Paris"];
  distances[to + size*from] = 506;

  to    = cityIndex["Marseille"];
  distances[to + size*from] = 951;

  to    = cityIndex["Bordeaux"];
  distances[to + size*from] = 496;

  to    = cityIndex["Toulouse"];
  distances[to + size*from] = 703;

  to    = cityIndex["Nantes"];
  distances[to + size*from] = 255;

  to    = cityIndex["Lille"];
  distances[to + size*from] = 599;

  to    = cityIndex["Metz"];
  distances[to + size*from] = 786;

  to    = cityIndex["Nancy"];
  distances[to + size*from] = 787;

  to    = cityIndex["Lyon"];
  distances[to + size*from] = 764;

  to    = cityIndex["Clermont-Ferrand"];
  distances[to + size*from] = 643;

  to    = cityIndex["Strasbourg"];
  distances[to + size*from] = 902;

  to    = cityIndex["Tours"];
  distances[to + size*from] = 402;

  to    = cityIndex["Limoges"];
  distances[to + size*from] = 520;

  // ========================================================
  from = cityIndex["Nantes"];
  to   = cityIndex["Nantes"];
  distances[to + size*from] = 0;
  // ========================================================

  to    = cityIndex["Paris"];
  distances[to + size*from] = 343;

  to    = cityIndex["Marseille"];
  distances[to + size*from] = 696;

  to    = cityIndex["Bordeaux"];
  distances[to + size*from] = 276;

  to    = cityIndex["Toulouse"];
  distances[to + size*from] = 466;

  to    = cityIndex["Brest"];
  distances[to + size*from] = 255;

  to    = cityIndex["Lille"];
  distances[to + size*from] = 508;

  to    = cityIndex["Metz"];
  distances[to + size*from] = 612;

  to    = cityIndex["Nancy"];
  distances[to + size*from] = 600;

  to    = cityIndex["Lyon"];
  distances[to + size*from] = 516;

  to    = cityIndex["Clermont-Ferrand"];
  distances[to + size*from] = 390;

  to    = cityIndex["Strasbourg"];
  distances[to + size*from] = 710;

  to    = cityIndex["Tours"];
  distances[to + size*from] = 170;

  to    = cityIndex["Limoges"];
  distances[to + size*from] = 265;

  // ========================================================
  from = cityIndex["Lille"];
  to   = cityIndex["Lille"];
  distances[to + size*from] = 0;
  // ========================================================

  to    = cityIndex["Paris"];
  distances[to + size*from] = 204;

  to    = cityIndex["Marseille"];
  distances[to + size*from] = 835;

  to    = cityIndex["Bordeaux"];
  distances[to + size*from] = 700;

  to    = cityIndex["Toulouse"];
  distances[to + size*from] = 792;

  to    = cityIndex["Brest"];
  distances[to + size*from] = 599;

  to    = cityIndex["Nantes"];
  distances[to + size*from] = 508;

  to    = cityIndex["Metz"];
  distances[to + size*from] = 280;

  to    = cityIndex["Nancy"];
  distances[to + size*from] = 312;

  to    = cityIndex["Lyon"];
  distances[to + size*from] = 558;

  to    = cityIndex["Clermont-Ferrand"];
  distances[to + size*from] = 540;

  to    = cityIndex["Strasbourg"];
  distances[to + size*from] = 408;

  to    = cityIndex["Tours"];
  distances[to + size*from] = 400;

  to    = cityIndex["Limoges"];
  distances[to + size*from] = 550;

  // ========================================================
  from = cityIndex["Metz"];
  to   = cityIndex["Metz"];
  distances[to + size*from] = 0;
  // ========================================================

  to    = cityIndex["Paris"];
  distances[to + size*from] = 281;

  to    = cityIndex["Marseille"];
  distances[to + size*from] = 651;

  to    = cityIndex["Bordeaux"];
  distances[to + size*from] = 700;

  to    = cityIndex["Toulouse"];
  distances[to + size*from] = 713;

  to    = cityIndex["Brest"];
  distances[to + size*from] = 786;

  to    = cityIndex["Nantes"];
  distances[to + size*from] = 612;

  to    = cityIndex["Lille"];
  distances[to + size*from] = 280;

  to    = cityIndex["Nancy"];
  distances[to + size*from] = 48;

  to    = cityIndex["Lyon"];
  distances[to + size*from] = 387;

  to    = cityIndex["Clermont-Ferrand"];
  distances[to + size*from] = 439;

  to    = cityIndex["Strasbourg"];
  distances[to + size*from] = 130;

  to    = cityIndex["Tours"];
  distances[to + size*from] = 450;

  to    = cityIndex["Limoges"];
  distances[to + size*from] = 520;

  // ========================================================
  from = cityIndex["Nancy"];
  to   = cityIndex["Nancy"];
  distances[to + size*from] = 0;
  // ========================================================

  to    = cityIndex["Paris"];
  distances[to + size*from] = 282;

  to    = cityIndex["Marseille"];
  distances[to + size*from] = 604;

  to    = cityIndex["Bordeaux"];
  distances[to + size*from] = 671;

  to    = cityIndex["Toulouse"];
  distances[to + size*from] = 674;

  to    = cityIndex["Brest"];
  distances[to + size*from] = 787;

  to    = cityIndex["Nantes"];
  distances[to + size*from] = 600;

  to    = cityIndex["Lille"];
  distances[to + size*from] = 312;

  to    = cityIndex["Metz"];
  distances[to + size*from] = 48;

  to    = cityIndex["Lyon"];
  distances[to + size*from] = 342;

  to    = cityIndex["Clermont-Ferrand"];
  distances[to + size*from] = 400;

  to    = cityIndex["Strasbourg"];
  distances[to + size*from] = 116;

  to    = cityIndex["Tours"];
  distances[to + size*from] = 434;

  to    = cityIndex["Limoges"];
  distances[to + size*from] = 489;

  // ========================================================
  from = cityIndex["Lyon"];
  to   = cityIndex["Lyon"];
  distances[to + size*from] = 0;
  // ========================================================

  to    = cityIndex["Paris"];
  distances[to + size*from] = 392;

  to    = cityIndex["Marseille"];
  distances[to + size*from] = 278;

  to    = cityIndex["Bordeaux"];
  distances[to + size*from] = 436;

  to    = cityIndex["Toulouse"];
  distances[to + size*from] = 360;

  to    = cityIndex["Brest"];
  distances[to + size*from] = 764;

  to    = cityIndex["Nantes"];
  distances[to + size*from] = 516;

  to    = cityIndex["Lille"];
  distances[to + size*from] = 558;

  to    = cityIndex["Metz"];
  distances[to + size*from] = 387;

  to    = cityIndex["Nancy"];
  distances[to + size*from] = 342;

  to    = cityIndex["Clermont-Ferrand"];
  distances[to + size*from] = 136;

  to    = cityIndex["Strasbourg"];
  distances[to + size*from] = 384;

  to    = cityIndex["Tours"];
  distances[to + size*from] = 366;

  to    = cityIndex["Limoges"];
  distances[to + size*from] = 278;

  // ========================================================
  from = cityIndex["Clermont-Ferrand"];
  to   = cityIndex["Clermont-Ferrand"];
  distances[to + size*from] = 0;
  // ========================================================

  to    = cityIndex["Paris"];
  distances[to + size*from] = 347;

  to    = cityIndex["Marseille"];
  distances[to + size*from] = 330;

  to    = cityIndex["Bordeaux"];
  distances[to + size*from] = 306;

  to    = cityIndex["Toulouse"];
  distances[to + size*from] = 275;

  to    = cityIndex["Brest"];
  distances[to + size*from] = 643;

  to    = cityIndex["Nantes"];
  distances[to + size*from] = 390;

  to    = cityIndex["Lille"];
  distances[to + size*from] = 540;

  to    = cityIndex["Metz"];
  distances[to + size*from] = 439;

  to    = cityIndex["Nancy"];
  distances[to + size*from] = 400;

  to    = cityIndex["Lyon"];
  distances[to + size*from] = 136;

  to    = cityIndex["Strasbourg"];
  distances[to + size*from] = 471;

  to    = cityIndex["Tours"];
  distances[to + size*from] = 257;

  to    = cityIndex["Limoges"];
  distances[to + size*from] = 142;

  // ========================================================
  from = cityIndex["Strasbourg"];
  to   = cityIndex["Strasbourg"];
  distances[to + size*from] = 0;
  // ========================================================

  to    = cityIndex["Paris"];
  distances[to + size*from] = 397;

  to    = cityIndex["Marseille"];
  distances[to + size*from] = 617;

  to    = cityIndex["Bordeaux"];
  distances[to + size*from] = 760;

  to    = cityIndex["Toulouse"];
  distances[to + size*from] = 737;

  to    = cityIndex["Brest"];
  distances[to + size*from] = 902;

  to    = cityIndex["Nantes"];
  distances[to + size*from] = 710;

  to    = cityIndex["Lille"];
  distances[to + size*from] = 408;

  to    = cityIndex["Metz"];
  distances[to + size*from] = 130;

  to    = cityIndex["Nancy"];
  distances[to + size*from] = 116;

  to    = cityIndex["Lyon"];
  distances[to + size*from] = 384;

  to    = cityIndex["Clermont-Ferrand"];
  distances[to + size*from] = 471;

  to    = cityIndex["Tours"];
  distances[to + size*from] = 543;

  to    = cityIndex["Limoges"];
  distances[to + size*from] = 578;

  // ========================================================
  from = cityIndex["Tours"];
  to   = cityIndex["Tours"];
  distances[to + size*from] = 0;
  // ========================================================

  to    = cityIndex["Paris"];
  distances[to + size*from] = 205;

  to    = cityIndex["Marseille"];
  distances[to + size*from] = 585;

  to    = cityIndex["Bordeaux"];
  distances[to + size*from] = 301;

  to    = cityIndex["Toulouse"];
  distances[to + size*from] = 426;

  to    = cityIndex["Brest"];
  distances[to + size*from] = 402;

  to    = cityIndex["Nantes"];
  distances[to + size*from] = 170;

  to    = cityIndex["Lille"];
  distances[to + size*from] = 400;

  to    = cityIndex["Metz"];
  distances[to + size*from] = 450;

  to    = cityIndex["Nancy"];
  distances[to + size*from] = 434;

  to    = cityIndex["Lyon"];
  distances[to + size*from] = 366;

  to    = cityIndex["Clermont-Ferrand"];
  distances[to + size*from] = 257;

  to    = cityIndex["Strasbourg"];
  distances[to + size*from] = 543;

  to    = cityIndex["Limoges"];
  distances[to + size*from] = 179;

  // ========================================================
  from = cityIndex["Limoges"];
  to   = cityIndex["Limoges"];
  distances[to + size*from] = 0;
  // ========================================================

  to    = cityIndex["Paris"];
  distances[to + size*from] = 347;

  to    = cityIndex["Marseille"];
  distances[to + size*from] = 431;

  to    = cityIndex["Bordeaux"];
  distances[to + size*from] = 182;

  to    = cityIndex["Toulouse"];
  distances[to + size*from] = 249;

  to    = cityIndex["Brest"];
  distances[to + size*from] = 520;

  to    = cityIndex["Nantes"];
  distances[to + size*from] = 265;

  to    = cityIndex["Lille"];
  distances[to + size*from] = 550;

  to    = cityIndex["Metz"];
  distances[to + size*from] = 520;

  to    = cityIndex["Nancy"];
  distances[to + size*from] = 489;

  to    = cityIndex["Lyon"];
  distances[to + size*from] = 278;

  to    = cityIndex["Clermont-Ferrand"];
  distances[to + size*from] = 142;

  to    = cityIndex["Strasbourg"];
  distances[to + size*from] = 578;

  to    = cityIndex["Tours"];
  distances[to + size*from] = 179;

  return distances;

} // init_distance_matrix

// ==================================================
// ==================================================
//! extract a smaller matrix for debug purpose
std::vector<int> init_distance_matrix_small(int n)
{

  auto cityIndex = makeCityMap();
  auto size = cityIndex.size();

  auto distances = init_distance_matrix();

  std::vector<int> distances_small(n*n);

  for (int from=0; from<n; ++from)
  {
    for (int to=0; to<n; ++to)
    {

      distances_small[to + n*from] = distances[to+size*from];

    }
  }

  return distances_small;

} // init_distance_matrix_small

#endif // TSP_H_
