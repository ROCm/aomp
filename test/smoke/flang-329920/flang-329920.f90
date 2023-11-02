module mpas_dmpar

    type field2DReal
        real, dimension(:,:), pointer :: array => null()
    end type field2DReal
    type field3DReal
        real, dimension(:,:,:), pointer :: array => null()
    end type field3DReal

    interface mpas_dmpar_exch_halo_acc
        module procedure mpas_dmpar_exch_halo_2d_acc
        module procedure mpas_dmpar_exch_halo_3d_acc
    end interface mpas_dmpar_exch_halo_acc

contains

    subroutine mpas_dmpar_exch_halo_2d_acc(field)
        implicit none
        type(field2DReal),pointer :: field
        real, dimension(:,:), pointer :: array => null()
        array => field % array
        !$omp target enter data map(to:array)
    end subroutine mpas_dmpar_exch_halo_2d_acc

    subroutine mpas_dmpar_exch_halo_3d_acc(field)
        implicit none
        type(field3DReal),pointer :: field
        real, dimension(:,:,:), pointer :: array1 => null()
        array1 => field % array
        !$omp target enter data map(to:array1)
    end subroutine mpas_dmpar_exch_halo_3d_acc

end module mpas_dmpar

program mpas_driver
    use mpas_dmpar
    type(field2DReal), pointer :: field2D
    type(field3DReal), pointer :: field3D
    allocate(field2D)
    allocate(field3D)
    allocate(field2D % array(26,21178))
    allocate(field3D % array(1,26,21178))
    write(*,*) "ompt map to 2D array"
    call mpas_dmpar_exch_halo_acc(field2D)
    write(*,*) "ompt map to 3D array"
    call mpas_dmpar_exch_halo_acc(field3D)
    print *, "PASS"
    return
end program mpas_driver
