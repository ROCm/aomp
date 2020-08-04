program foo
  integer, allocatable:: a(:)
  integer, target:: b(100)
  integer, pointer::c(:)
  allocate(a(100))
  c=>b
  c = 100
  a = 4
!$omp target 
  a(50) = b(50) + 2
  a(49) = c(49) + 1
  b(48) = 0
  c(47) = 99
!$omp end target
  write(6,*) a(48), a(49), a(50)
  write(6,*) b(47), b(48)
  write(6,*) c(46), c(47)
  if (a(48).eq.4 .and. a(49).eq.101 .and. a(50).eq.102 .and. b(47).eq.99 .and. &
& b(48).eq. 0 .and. c(46).eq.100 .and. c(47).eq.99) then
     write(6,*) "passed"
   else
     write(6,*) "Failed!!!"
   endif
end
