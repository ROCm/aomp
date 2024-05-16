module allocator

    interface
        subroutine hipmalloc_(ptr, nsize) bind(C, name="hipMalloc")
            use iso_c_binding 
            implicit none
            type(c_ptr) :: ptr
            integer(c_size_t), value :: nsize
        end subroutine

        function omp_target_alloc(nsize, device_num) bind(C, name="omp_target_alloc")
            use iso_c_binding 
            implicit none
            type(c_ptr) :: omp_target_alloc
            integer(c_size_t), value :: nsize
            integer(c_int), value :: device_num
        end function
    end interface

    contains

    subroutine hipmalloc(a, n) 
            use iso_c_binding 
            implicit none
            double precision, dimension(:), pointer, intent(in) :: a
            integer, intent(in)   :: n
            double precision :: elem
            type(c_ptr) :: ptr
            integer(c_size_t) :: nsize

            nsize = n * sizeof(elem)

            call hipmalloc_(ptr, nsize)
            call c_f_pointer(ptr, a, [n])
    end subroutine

    subroutine ompmalloc(a, n, device_num) 
            use iso_c_binding 
            implicit none
            double precision, dimension(:), pointer, intent(in) :: a
            integer, intent(in)   :: n
            double precision :: elem
            type(c_ptr) :: ptr
            integer(c_int) :: device_num
            integer(c_size_t) :: nsize

            nsize = n * sizeof(elem)

            ptr =  omp_target_alloc(nsize, device_num)
            call c_f_pointer(ptr, a, [n])
    end subroutine


end module 
