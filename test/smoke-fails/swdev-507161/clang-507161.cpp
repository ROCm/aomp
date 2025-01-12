
template<typename T>
class A {
public:
  //- Construct null with invalid (-1) for distance, null constructor
  //  for data
  inline A();
  
  //- Construct from distance, data
  inline A
  (
   const int distance,
   const T& data
   );

  inline int getDistance() {
    return distance;
  }
  
  void updateB(A &updateInfo);

protected:
  int distance;
  T _data;
};

class B {
public:
  int b;
};

template<typename T>
A<T>::A() {
  distance = -1;
}

template<typename T>
void A<T>::updateB(A<T> &updateInfo) {
  this->operator=(updateInfo);
}

int main() {
  A<B>a;

  A<B>b;
  a.updateB(b);
  if (a.getDistance() == -1) return 0 ;
  return 1;

}
