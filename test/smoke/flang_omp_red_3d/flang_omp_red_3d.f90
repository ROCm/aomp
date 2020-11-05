program TargetReduction_3D

    use OMP_LIB
    implicit none
    
    integer :: &
      iV, jV, kV, &
      StopCode
    integer, dimension ( 3 ) :: &
      lV, uV, oV
    real, dimension ( :, :, : ), allocatable :: &
      FEP_1, FEP_2, FEP_3, &
      FEM_1, FEM_2, FEM_3 
    real :: &
      MaxSpeed
    real, parameter :: &
      DEFINED_MAX_SPEED = 10.0
      
    StopCode = 0
    
    allocate ( FEP_1 ( 32, 16, 8 ) )
    allocate ( FEP_2 ( 32, 16, 8 ) )
    allocate ( FEP_3 ( 32, 16, 8 ) )
    allocate ( FEM_1 ( 32, 16, 8 ) )
    allocate ( FEM_2 ( 32, 16, 8 ) )
    allocate ( FEM_3 ( 32, 16, 8 ) )
    
    call random_number ( FEP_1 )
    call random_number ( FEP_2 )
    call random_number ( FEP_3 )
    call random_number ( FEM_1 )
    call random_number ( FEM_2 )
    call random_number ( FEM_3 )
    
    lV = lbound ( FEP_1 )
    uV = ubound ( FEP_1 )
        
    FEM_1 = -1.0 * FEM_1
    FEM_2 = -1.0 * FEM_2
    FEM_3 = -1.0 * FEM_3
    
    FEP_1 ( 15, 15, 4 ) = DEFINED_MAX_SPEED
    
    !$OMP  target enter data &
    !$OMP  map ( to: FEP_1, FEP_2, FEP_3, FEM_1, FEM_2, FEM_3 )

    MaxSpeed = - huge ( 1.0 )

    print*, 'MaxSpeed init',  MaxSpeed 
    print*, 'Num devices: ', omp_get_num_devices()
    print*, 'Expected MaxSpeed: ', DEFINED_MAX_SPEED

    !$OMP  target teams distribute parallel do simd collapse ( 3 ) &
    !$OMP  schedule ( static, 1 ) private ( iV, jV, kV ) &
    !$OMP  reduction ( max : MaxSpeed ) 
    do kV = lV ( 3 ), uV ( 3 )   
      do jV = lV ( 2 ), uV ( 2 )  
        do iV = lV ( 1 ), uV ( 1 )
          MaxSpeed &
            = max (  FEP_1 ( iV, jV, kV ),  FEP_2 ( iV, jV, kV ), & 
                     FEP_3 ( iV, jV, kV ), -FEM_1 ( iV, jV, kV ), & 
                    -FEM_2 ( iV, jV, kV ), -FEM_3 ( iV, jV, kV ), MaxSpeed  )
        end do
      end do
    end do
    !$OMP  end target teams distribute parallel do simd

    print*, 'MaxSpeed reduced on device ', MaxSpeed
    if ( MaxSpeed /= DEFINED_MAX_SPEED ) then
      print*, 'Reduction on device: FAILED'
      StopCode = 1
    end if
    
    MaxSpeed = - huge ( 1.0 )
    MaxSpeed = max ( maxval (  FEP_1 ), maxval (  FEP_2 ), &
                     maxval (  FEP_3 ), maxval ( -FEM_1 ), &           
                     maxval ( -FEM_2 ), maxval ( -FEM_3 ), MaxSpeed  )
    
    print*, 'MaxSpeed reduced on host ', MaxSpeed
    if ( MaxSpeed /= DEFINED_MAX_SPEED ) then
      print*, 'Reduction on host: FAILED'
      StopCode = 1 
    end if
    
    if ( StopCode /= 0 ) stop StopCode
    
end program TargetReduction_3D

