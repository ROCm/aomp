module KIND_BIG_Singleton

  type, public :: KindBigSingleton
    integer :: &
      CHARACTER  = selected_char_kind ( 'ASCII' ), &
      SUPERSCRIPT_MINUS = int ( z'003F' ), &
      SUPERSCRIPT_1     = int ( z'003F' )
  end type

  type ( KindBigSingleton ), public, parameter :: &
    KIND_BIG = KindBigSingleton ( )

  integer, public, parameter :: &
    KBCH = KIND_BIG % CHARACTER

end module KIND_BIG_Singleton

module UNIT_Singleton

  use KIND_BIG_Singleton, &
        KB => KIND_BIG

  implicit none
  private

    character ( 5, KBCH ) :: &
      MeV_Minus_1_KBCH &
        = 'MeV' // char ( KB % SUPERSCRIPT_MINUS, KBCH ) &
                     // char ( KB % SUPERSCRIPT_1, KBCH )

end module UNIT_Singleton
program computers
  print *,'hello computers'
end program
