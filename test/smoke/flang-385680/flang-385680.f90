#define my_sin(val) sin(val)

program main

  integer,parameter::rtype=8

  real(rtype)::x

  x=1._rtype

  x=1._rtype+my_sin(x)

  !workaround

  !x=1._rtype+ my_sin(x)

  print*,x

end program
