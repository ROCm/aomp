program main
  REAL :: inc_r = 0.5
  REAL :: inc_w = 0.5
  !$omp target map(tofrom:inc_w)
      inc_w = inc_r + inc_w
      inc_r = 15
  !$omp end target    
  
PRINT *, inc_r ! should return 0.5/.5 the fortran equivelant, it's a bycopy with no map
PRINT *, inc_w ! should return 1.0/1. the fortran equivelant, it's a byref with a map


if (inc_r /= 0.5 .OR. inc_w /= 1.0) then     
  print*, "======= FORTRAN Test Failed! ======="
  stop 1    
end if  

print*, "======= FORTRAN Test passed! ======="

end program
