program main
  INTEGER :: sp_write(10) = (/0,0,0,0,0,0,0,0,0,0/)

  !$omp target map(tofrom:sp_write(1:10))
      do index = 1, 10
          sp_write(index) = index
      end do
  !$omp end target

  do i = 1, 10
      if (sp_write(i) /= i) then
          print*, "======= FORTRAN Test Failed! ======="
          stop 1    
      end if  
  end do 

  print*, "======= FORTRAN Test passed! ======="
  
end program
