#include <stdio.h>

// link are not supported in the compiler & runtime at this time
#define TEST_LINK  0

//
// All in a single declare target region.
//
#pragma omp declare target
int VA1;
int VA2 = 2;
int VA3, VA4 = 4;
int VA5 = 5, VA6 = 6;

void FA1(int &v) {
  ++v;
}

template<typename T>
void FA2(T &v) {
  ++v;
}

struct SA1 {
  int var;
  void f(int v) {
    var += v;
  }
  SA1(int v) : var(v+1) {}
};

template<typename T>
struct SA2 {
  T var;
  void f(T v) {
    var += v;
  }
  SA2(T v) : var(v+1) {}
};

SA1 VA10(123);
SA2<int> VA20(456);
#pragma omp end declare target

//
// All in an implicit 'to' list
//
int VB1;
int VB2 = 2;
int VB3, VB4 = 4;
int VB5 = 5, VB6 = 6;

void FB1(int &v) {
  ++v;
}

#pragma omp declare target
struct SB1 {
  int var;
  void f(int v) {
    var += v;
  }
  SB1(int v) : var(v+1) {}
};

template<typename T>
struct SB2 {
  T var;
  void f(T v) {
    var += v;
  }
  SB2(T v) : var(v+1) {}
};
#pragma omp end declare target

SB1 VB10(123);
SB2<int> VB20(456);
#pragma omp declare target (VB1, VB2, VB3, VB4, VB5, VB6, FB1, VB10, VB20)

//
// All in a 'to' list
//
int VC1;
int VC2 = 2;
int VC3, VC4 = 4;
int VC5 = 5, VC6 = 6;

void FC1(int &v) {
  ++v;
}

#pragma omp declare target
struct SC1 {
  int var;
  void f(int v) {
    var += v;
  }
  SC1(int v) : var(v+1) {}
};

template<typename T>
struct SC2 {
  T var;
  void f(T v) {
    var += v;
  }
  SC2(T v) : var(v+1) {}
};
#pragma omp end declare target

SC1 VC10(123);
SC2<int> VC20(456);
#pragma omp declare target to(VC1, VC2, VC3, VC4, VC5, VC6, FC1, VC10, VC20)

//
// All in a 'link' list
//
int VD1;
int VD2 = 2;
int VD3, VD4 = 4;
int VD5 = 5, VD6 = 6;

void FD1(int &v) {
  ++v;
}

#pragma omp declare target
struct SD1 {
  int var;
  void f(int v) {
    var += v;
  }
  SD1(int v) : var(v+1) {}
};

template<typename T>
struct SD2 {
  T var;
  void f(T v) {
    var += v;
  }
  SD2(T v) : var(v+1) {}
};
#pragma omp end declare target

SD1 VD10(123);
SD2<int> VD20(456);

#if TEST_LINK
  #pragma omp declare target link(VD1, VD2, VD3, VD4, VD5, VD6, FD1, VD10, VD20)
#else
  #pragma omp declare target to(VD1, VD2, VD3, VD4, VD5, VD6, FD1, VD10, VD20)
#endif
//
// Declaration in declare target but definition shows up later.
//

#pragma omp declare target 
void FE1(int &v);
template<typename T>
void FE2(T &v);
#pragma omp end declare target 

void FF1(int &v);
#pragma omp declare target (FF1)

void FG1(int &v);
#pragma omp declare target to(FG1)

void FH1(int &v);
#pragma omp declare target link(FH1)

//
// Declaration has no declare target, only definition has.
//
void FI1(int &v);
template<typename T>
void FI2(T &v);

void FJ1(int &v);

void FK1(int &v);

void FL1(int &v);

int main(int argc, char * argv[]) {
  
  int RA, RB, RC, RD, R=0;

  #pragma omp target map(from:RA, RB, RC, RD) map(R)
  {
    int b,c;
    
    VA1 = VA2 + 1;
    VA3 = 2*VA1;
    RA = VA6 * VA5 + VA4 + VA3;
    FA1(RA);
    FA2(RA);
    VA10.f(RA);
    RA += VA10.var;
    VA20.f(RA);
    RA += VA20.var;
    
    VB1 = VB2 + 1;
    VB3 = 2*VB1;
    RB = VB6 * VB5 + VB4 + VB3;
    FB1(RB);
    FB1(RB);
    VB10.f(RB);
    RB += VB10.var;
    VB20.f(RB);
    RB += VB20.var;
    
    VC1 = VC2 + 1;
    VC3 = 2*VC1;
    RC = VC6 * VC5 + VC4 + VC3;
    FC1(RC);
    FC1(RC);
    VC10.f(RC);
    RC += VC10.var;
    VC20.f(RC);
    RC += VC20.var;
    
    VD1 = VD2 + 1;
    VD3 = 2*VD1;
    RD = VD6 * VD5 + VD4 + VD3;
    FD1(RD);
    FD1(RD);
    VD10.f(RD);
    RD += VD10.var;
    VD20.f(RD);
    RD += VD20.var;

    FE1(R);
    FF1(R);
    FG1(R);
    FH1(R);
    FI1(R);
    FI2(R);
    FJ1(R);
    FK1(R);
    FL1(R);
  }
  
  printf("RA = %d(873), RB = %d(873), RC = %d(873), RD = %d(873), R = %d(9)\n", RA, RB, RC, RD, R);
  return 0;
}


void FE1(int &v) {
  ++v;
}

template<typename T>
void FE2(T &v) {
  ++v;
}

void FF1(int &v) {
  ++v;
}

void FG1(int &v) {
  ++v;
}

void FH1(int &v) {
  ++v;
}

#pragma omp declare target
void FI1(int &v) {
  ++v;
}
template<typename T>
void FI2(T &v) {
  ++v;
}
#pragma omp end declare target

void FJ1(int &v) {
  ++v;
}
#pragma omp declare target (FJ1)

void FK1(int &v) {
  ++v;
}
#pragma omp declare target to(FK1)

void FL1(int &v) {
  ++v;
}
#pragma omp declare target link(FL1)
