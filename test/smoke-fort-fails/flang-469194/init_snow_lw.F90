SUBROUTINE INIT_SNOW_LW(TPSNOW)
  use MODE_SNOW3L
  implicit none

TYPE SURF_SNOW
CHARACTER(LEN=3)                :: SCHEME    ! snow scheme used
INTEGER                         :: NLAYER    ! number of layers
REAL, DIMENSION(:,:), POINTER :: DEPTH
REAL, DIMENSION(:,:), POINTER :: WSNOW       ! snow (& liq. water) content (kg/m2)
REAL, DIMENSION(:,:), POINTER :: HEAT        ! heat content                (J/m2)
REAL, DIMENSION(:,:), POINTER :: T           ! temperature '1-L'
REAL, DIMENSION(:,:), POINTER :: TEMP        ! temperature '3-L' (K)
REAL, DIMENSION(:,:), POINTER :: RHO         ! density
REAL, DIMENSION(:)  , POINTER :: ALB         ! snow surface albedo
REAL, DIMENSION(:)  , POINTER :: ALBVIS      ! snow surface visible albedo
REAL, DIMENSION(:)  , POINTER :: ALBNIR      ! snow surface near-infrared albedo
REAL, DIMENSION(:)  , POINTER :: ALBFIR      ! snow surface far-infrared albedo
REAL, DIMENSION(:)  , POINTER :: EMIS        ! snow surface emissivity
REAL, DIMENSION(:)  , POINTER :: TS          ! snow surface temperature
REAL, DIMENSION(:,:), POINTER :: GRAN1       ! snow grain parameter 1
REAL, DIMENSION(:,:), POINTER :: GRAN2       ! snow grain parameter 2
REAL, DIMENSION(:,:), POINTER :: HIST        ! snow historical variable (non dendritic case)
REAL, DIMENSION(:,:), POINTER :: AGE         ! snow grain age
REAL, DIMENSION(:)  , POINTER :: DEP_SUP     ! snow depth in superior profile
REAL, DIMENSION(:)  , POINTER :: DEP_TOT     ! total snow depth (m)
REAL, DIMENSION(:)  , POINTER :: DEP_HUM     ! height of the uppest continuous block of humid snow in the sup
REAL, DIMENSION(:)  , POINTER :: NAT_LEV     ! natural risk index (0-6)
REAL, DIMENSION(:)  , POINTER :: PRO_SUP_TYP ! type of superior profile (0, 4, 5, 6)
REAL, DIMENSION(:)  , POINTER :: AVA_TYP     ! type of avalanche (0-6)
REAL, DIMENSION(:,:,:), POINTER :: IMPUR ! impurity content with new dim
END TYPE SURF_SNOW

  TYPE(SURF_SNOW),      INTENT(INOUT) :: TPSNOW  ! snow characteristics
  REAL, DIMENSION(1000) :: TS
  REAL :: XLMTT
  REAL :: XTT
  REAL :: XUNDEF

  XTT = 0
  XLMTT = 0
  XUNDEF = 0

  WHERE(TPSNOW%WSNOW(:,1)==0)
    TS (:)= XUNDEF
  ELSEWHERE
    TS(:) = XTT + (TPSNOW%HEAT(:,1) + XLMTT * TPSNOW%RHO(:,1)) / SNOW3LSCAP(TPSNOW%RHO(:,1))
  END WHERE
END SUBROUTINE INIT_SNOW_LW
