program test
    use omp_lib
    use f90deviceio
    write(0, *) "Before target "
    !$omp target
       call f90print("In tagret ")
    !$omp end target
    write(0,*) "After target "
end program test
