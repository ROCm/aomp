module PolytropicFluid_Form
  implicit none
    interface
      module subroutine ComputeConservedKernel &
                   ( G, E, N, V_1, V_2, V_3, UseDevice )
        implicit none
        real ( 8 ), dimension ( : ), intent ( inout ) :: &
          G
        real ( 8 ), dimension ( : ), intent ( in ) :: &
          E, &
          N, &
          V_1, V_2, V_3
        logical ( 1 ), intent ( in ) :: &
          UseDevice
      end subroutine ComputeConservedKernel
    end interface
end module PolytropicFluid_Form

submodule ( PolytropicFluid_Form ) PolytropicFluid_Kernel
  implicit none
contains
  module procedure ComputeConservedKernel
    integer ( 4 ) :: &
      iV
    if ( UseDevice ) then
      !$OMP target teams distribute parallel do simd &
      !$OMP schedule ( auto )
      do iV = 1, size ( G )
        G ( iV ) = E ( iV ) + 0.5_8 * N ( iV ) &
                   * ( V_1 ( iV ) * V_1 ( iV ) &
                       + V_2 ( iV ) * V_2 ( iV ) &
                       + V_3 ( iV ) * V_3 ( iV ) )
      end do
      !$OMP end target teams distribute parallel do simd
    else
      !$OMP parallel do simd &
      !$OMP schedule ( runtime )
      do iV = 1, size ( G )
        G ( iV ) = E ( iV ) + 0.5_8 * N ( iV ) &
                   * ( V_1 ( iV ) * V_1 ( iV ) &
                       + V_2 ( iV ) * V_2 ( iV ) &
                       + V_3 ( iV ) * V_3 ( iV ) )
      end do
      !$OMP end parallel do simd
    end if
  end procedure ComputeConservedKernel
end submodule PolytropicFluid_Kernel
