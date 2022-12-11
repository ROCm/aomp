#pragma once

#define __HD__

template<class T>
class ADD{
  public:

  __HD__ static inline T OP(T& v1, T& v2){
      return v1 + v2;
  }

  __HD__ static inline T OP(const T& v1, const T& v2){
      return v1 + v2;
  }

  template<class P>
  __HD__ static bool validate(T val, P val1){
    P tmp = -val1/2;
    if ( val == ((val1-1)%2) * tmp )
      return true;
    return false;
  }
  template<class P>
  __HD__ static T correct(P val1){
    P tmp = -val1/2;
    return ((val1-1)%2) * tmp ;
  }

  __HD__ static T init(){
    return 0;
  }

  __HD__ static T init(long i, long elements){
    return static_cast<T>(i  - (elements/2));
  }

  __HD__  static const char *info(){
    return "add";
  }
};

