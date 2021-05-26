program test
    real :: a = 1.0
    real :: b = 1.0
    real :: at,bt

#define OFFLOAD_ATAN 1
#ifdef OFFLOAD_ATAN
!$omp target map(tofrom: a,b,at,bt)
#endif
    at = atan(a,b)
    bt = atan2(a,b)
#ifdef OFFLOAD_ATAN
!$omp end target
#endif
    
    print *, "atan(", a, ",", b, ")=", at
    print *, "atan(", a, ",", b, ")=", bt
end program test
