program target_update

 

  real, dimension ( 10 ) :: &

    A

    

  !$omp target enter data map ( alloc : A )

  call random_number ( A )

  !$omp target update to(A) nowait

 

end program target_update
