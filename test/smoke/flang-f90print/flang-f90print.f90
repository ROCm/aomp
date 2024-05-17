program test
    use omp_lib
    use f90deviceio
    write(0, *) "Before target "
    !$omp target
       call f90print("f90print inside target region")
       !if the above line is not printed then the test has failed
    !$omp end target
    write(0,*) "After target "
end program test
