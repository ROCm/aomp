program test
    real,pointer :: a(:),b(:)
    allocate(a(10),b(10))
    b = 1000
    !$omp target enter data map(alloc:a,b)
    !$omp target teams distribute parallel do private(k)
       do  k = 1, 10
                a(k) = 0.0d0
                b(k) = 0.0d0
       end do
    !$omp end target teams distribute parallel do
    write(*,*) "Work Done"
    return
end program test
