program bug
    real, dimension(*) :: pointee
    pointer (ptr, pointee)
end program bug

subroutine bug2()
    real, dimension(*) :: pointee
    pointer (ptr, pointee)
end subroutine bug2
