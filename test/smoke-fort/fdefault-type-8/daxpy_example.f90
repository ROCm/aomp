program daxpy_example
 
    !use omp_lib
    use iso_c_binding
    implicit none
    real*8 :: a, b                ! Scalar multiplier
    real*8 :: c
    real*8 :: mysum
 
    a = 1.0
    b = 2.0d-1
    !integer :: get_cpu_core
 
    c = mysum(a,2.0d-1)
    write(*,*) "c = ", c
 
    c = mysum(a,b)
    write(*,*) "c = ", c
 
 
end program daxpy_example
 
 
real(8) function mysum(a,b)
        implicit none
        real*8 :: a,b
        mysum = a+b
end function mysum
