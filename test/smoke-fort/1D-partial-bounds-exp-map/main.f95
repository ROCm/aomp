program main
  INTEGER :: sp_read(10) = (/1,2,3,4,5,6,7,8,9,10/)
  INTEGER :: sp_write(10) = (/0,0,0,0,0,0,0,0,0,0/)

  !$omp target map(tofrom:sp_read(2:5)) map(tofrom:sp_write(2:5))
      do i = 2, 5
          sp_write(i) = sp_read(i)
      end do
  !$omp end target

  if (sp_write(1) /= 0) then
      print*, "======= FORTRAN Test Failed! ======="
      stop 1    
  end if  

  do i = 2, 5
      if (sp_write(i) /= i) then
          print*, "======= FORTRAN Test Failed! ======="
          stop 1    
      end if  
  end do 
      
  do i = 6, 10
      if (sp_write(i) /= 0) then
          print*, "======= FORTRAN Test Failed! ======="
          stop 1    
      end if  
  end do 

  print*, "======= FORTRAN Test passed! ======="
  
end program
