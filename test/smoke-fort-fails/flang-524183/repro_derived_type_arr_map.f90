!************************************
PROGRAM repro_derived_type_arr_map
        implicit none
         TYPE  btype
                 REAL :: c
         END TYPE
         TYPE  atype
                 TYPE(btype),allocatable,dimension(:) :: b
         END TYPE
         TYPE(atype) :: a

        allocate(a%b(3))
        a%b(2)%c = 1.0
        !$omp target update to(a%b(2)%c)
END PROGRAM
!************************************
