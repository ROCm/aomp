module Jacobi_Form

  use omp_lib 
  use iso_fortran_env
  
  implicit none
  private
  
  integer :: &
    MAX_ITERATIONS = 10000
  real ( real64 ), parameter :: &
    MAX_RESIDUAL = 1e-5_real64
    
  public :: &
    Initialize, &
    Compute, &
    Compute_OpenMP_GPU_Teams, &
    Validate, &
    ShowResults

contains 


  subroutine Initialize ( T, T_Init, T_Reference, nCells )
    
    real ( real64 ), dimension ( :, : ), allocatable, intent ( out ) :: &
      T, &
      T_Init, &
      T_Reference
    integer, dimension ( : ), intent ( out ) :: &
      nCells
      
    integer :: &
      iS, &
      SeedSize
    integer, dimension ( : ), allocatable :: &
      Seed
    real ( real64 ) :: &
      TestRandom
    character ( 31 ) :: &
      ExecName, &
      nCellsString, &
      MaxIterationsString
    
      
    !-- Parse command line options
    call get_command_argument ( 0, Value = ExecName )
    
    if ( command_argument_count ( ) == 2 ) then
      call get_command_argument ( 1, nCellsString )
      read ( nCellsString, fmt = '( i7 )' ) nCells ( 1 )
      nCells ( 2 ) = nCells ( 1 )

      call get_command_argument ( 2, MaxIterationsString )
      read ( MaxIterationsString, fmt = '( i7 )' ) MAX_ITERATIONS
      
      print*, 'Executing      : ', ExecName
      print*, 'nCells         : ', nCells ( 1 ), nCells ( 2 )
      print*, 'Max iterations : ', MAX_ITERATIONS
      print*, ''
      
    else
      print*, 'Usage: ' // trim ( ExecName ) // ' <nCells> <MaxIterations>'
      print*, '  where <nCells> and <MaxIterations> are integers.'
    end if
    
    !-- End parsing 
      
    allocate ( T      ( 0 : nCells ( 1 ) + 1,  0 : nCells ( 2 ) + 1 ) )
    allocate ( T_Init ( 0 : nCells ( 1 ) + 1,  0 : nCells ( 2 ) + 1 ) )
    allocate ( T_Reference ( 0 : nCells ( 1 ) + 1,  0 : nCells ( 2 ) + 1 ) )
    
    !-- Initialize seed for random number
    call random_seed ( size = SeedSize )
    allocate ( Seed ( SeedSize ) )
    do iS = 1, SeedSize
        call system_clock ( Seed ( iS ), count_rate = TestRandom )
    end do
    call random_seed ( put = Seed )
    
    !-- Initialize T with random number
    call random_number ( T )
    T_Init = T
         
  end subroutine Initialize
  
  
  subroutine Compute ( T, nCells, Residual, nIterations )
    
    real ( real64 ), dimension ( 0 :, 0 : ), intent ( inout ) :: &
      T
    integer, dimension ( : ), intent ( in ) :: &
      nCells
    real ( real64 ), intent ( out ) :: &
      Residual
    integer, intent ( out ) :: &
      nIterations
    
    real ( real64 ) :: &
      TimeStart, &
      TimeTotal
    real ( real64 ), dimension ( :, : ), allocatable :: &
      T_New
    
    integer :: &
      iV, jV
      
    allocate ( T_New ( 0 : nCells ( 1 ) + 1,  0 : nCells ( 2 ) + 1 ) )
    
    nIterations = 0
    Residual    = huge ( 1.0_real64 )
    
    do while ( nIterations < MAX_ITERATIONS &
               .and. Residual  > MAX_RESIDUAL )
               
      do jV = 1, nCells ( 2 )
        do iV = 1, nCells ( 1 )
          T_New ( iV, jV ) &
            = 0.25 * (   T ( iV, jV - 1 ) + T ( iV, jV + 1 ) &
                       + T ( iV - 1, jV ) + T ( iV + 1, jV ) )
        end do
      end do 
      
      nIterations = nIterations + 1
      
      associate ( &
        T_P     => T     ( 1 : nCells ( 1 ), 1 : nCells ( 2 ) ), &
        T_New_P => T_New ( 1 : nCells ( 1 ), 1 : nCells ( 2 ) ) )
      
      Residual = maxval ( abs ( T_New_P - T_P ) )
      
      T_P = T_New_P
      
      end associate
      
    end do
    
  end subroutine Compute
 
 
  subroutine Compute_OpenMP_GPU_Teams &
               ( T, nCells, Residual, nIterations )
    
    real ( real64 ), dimension ( 0 :, 0 : ), intent ( inout ) :: &
      T
    integer, dimension ( : ), intent ( in ) :: &
      nCells
    real ( real64 ), intent ( out ) :: &
      Residual
    integer, intent ( out ) :: &
      nIterations
    
    real ( real64 ), dimension ( :, : ), allocatable :: &
      T_New
    
    integer :: &
      iV, jV
      
    allocate ( T_New ( 0 : nCells ( 1 ) + 1,  0 : nCells ( 2 ) + 1 ) )
    
    nIterations = 0
    Residual    = huge ( 1.0_real64 )
    
    do while ( nIterations < MAX_ITERATIONS &
               .and. Residual  > MAX_RESIDUAL )
      !$OMP target teams distribute collapse ( 2 ) &
      !$OMP map ( to: T ) map ( from: T_New )
      do jV = 1, nCells ( 2 )
        do iV = 1, nCells ( 1 )
          T_New ( iV, jV ) &
            = 0.25 * (   T ( iV, jV - 1 ) + T ( iV, jV + 1 ) &
                       + T ( iV - 1, jV ) + T ( iV + 1, jV ) )
        end do
      end do 
      !$OMP end target teams distribute 
      
      nIterations = nIterations + 1
      
      Residual = 0.0
      
      !$OMP target teams distribute collapse ( 2 ) &
      !$OMP   reduction ( max : Residual ) &
      !$OMP   map ( tofrom: T ) map ( to: T_New ) 
      do jV = 1, nCells ( 2 )
        do iV = 1, nCells ( 1 )
          Residual = max ( abs ( T_New ( iV, jV ) - T ( iV, jV ) ), Residual )
          T ( iV, jV ) = T_New ( iV, jV )
        end do
      end do 
      !$OMP end target teams distribute
    
    end do
    
  end subroutine Compute_OpenMP_GPU_Teams
  
  
  subroutine ShowResults &
               ( Description, Validation, Timing, Speedups, Residual, &
                 ValidationError, nIterations, CPU_nThreadsOptions )
    character ( * ), intent ( in ) :: &
      Description
    logical, intent ( in ) :: &
      Validation
    real ( real64 ), intent ( in ) :: &
      Timing, &
      Speedups, &
      Residual, &
      ValidationError
    integer, intent ( in ) :: &
      nIterations
    logical, intent ( in ), optional :: &
      CPU_nThreadsOptions
    
    character ( 6 ) :: &
      ValidationString
    logical :: &
      CPU_nThreads
      
    CPU_nThreads = .false.
    if ( present ( CPU_nThreadsOptions ) ) &
      CPU_nThreads = CPU_nThreadsOptions
    
    if ( Validation ) then
      ValidationString = 'PASSED'
    else
      ValidationString = 'FAILED'
    end if
    
    print '( a9, a, a9 )', &
          '======== ', trim ( Description ), ' ========='
    if ( CPU_nThreads ) &
      print '( a15, i3       )', 'nThreads      : ', omp_get_max_threads ( )
    print '( a15, es10.3e2 )', 'Timing (s)    : ', Timing
    print '( a15, es10.3e2 )', 'Speedups      : ', Speedups
    print '( a15, es10.3e2 )', 'Residual      : ', Residual
    print '( a15, i7       )', 'nIterations   : ', nIterations
    print '( a15, a8       )', 'Validation    : ', ValidationString
    if ( .not. Validation ) &
      print '( a15, es10.3e2 )', '  Error       : ', ValidationError
    print '( a )', ''
  
  end subroutine ShowResults
  
  
  subroutine Validate ( T1, T2, Validation, Error ) 
  
    real ( real64 ), dimension ( 0 :, 0 : ), intent ( in ) :: &
      T1, T2
    logical, intent ( out ) :: &
      Validation
    real ( real64 ), intent ( out ) :: &
      Error

    !-- Only compare the proper ( inner ) cells. 
    !   The following associate construct creates 'aliases' 
    !   that points to the proper cells in each dimension.
    associate ( &
      T1_P => T1 ( 1 : size ( T1, dim = 1 ) - 2, &
                   1 : size ( T1, dim = 2 ) - 2 ), &
      T2_P => T2 ( 1 : size ( T2, dim = 1 ) - 2, &
                   1 : size ( T2, dim = 2 ) - 2 ) )
    
    Error = maxval ( abs ( ( T1_P - T2_P ) / ( T1_P ) ) )
    
    if ( Error <= 20 * MAX_RESIDUAL ) then
      Validation = .true.
    else
      Validation = .false.
    end if
    
    end associate  !-- T1_P, T2_P
  
  end subroutine Validate 
  

end module Jacobi_Form


program Jacobi

  use omp_lib
  use iso_fortran_env
  use Jacobi_Form
  
  implicit none
  
  integer :: &
    nIterations
  integer, dimension ( 2 ) :: &
    nCells
  real ( real64 ) :: &
    TimeStart, &
    TimeTotal, &
    TimeSerial, &
    Residual, &
    ValidationError
  real ( real64 ), dimension ( :, : ), allocatable :: &
    T, &          !-- 
    T_Init, &     !-- A copy of initial condition
    T_Reference   !-- A copy of results from serial calculation
  logical :: &
    Validation
    
  nCells = -1
  
  call Initialize ( T, T_Init, T_Reference, nCells )
  if ( nCells ( 1 ) == -1 ) return
  
  TimeStart = omp_get_wtime ( )
  call Compute ( T, nCells, Residual, nIterations )
  TimeSerial = omp_get_wtime ( ) - TimeStart
  
  call ShowResults ( 'Serial', Validation = .true., &
                     Timing = TimeSerial, &
                     Speedups = 1.0_real64, &
                     Residual = Residual, &
                     ValidationError = 0.0_real64, &
                     nIterations = nIterations )
  
  !-- Save Results for other subroutine validation
  T_Reference = T
  
  
  !-- OpenMP_GPU_Teams

  T = T_Init
  
  TimeStart = omp_get_wtime ( )   
  call Compute_OpenMP_GPU_Teams &
         ( T, nCells, Residual, nIterations )
  TimeTotal = omp_get_wtime ( ) - TimeStart
  
  call Validate ( T, T_Reference, Validation, ValidationError )   

  call ShowResults &
         ( 'OpenMP_GPU_Teams', &
           Validation, TimeTotal, TimeSerial / TimeTotal, &
           Residual, ValidationError, nIterations )   

end program Jacobi
