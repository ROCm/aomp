program main
  use foo, only: foo_type
  implicit none
  type(foo_type) :: bar

  call bar%allocate(10)
  !$omp target enter data map(TO:bar)
  call bar%create_device(bar)
  call bar%delete_device(bar)  
  !$omp target exit data map(delete:bar)
  call bar%deallocate()
  
end program main
