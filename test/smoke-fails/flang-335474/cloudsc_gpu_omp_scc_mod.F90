! (C) Copyright 1988- ECMWF.
!
! This software is licensed under the terms of the Apache Licence Version 2.0
! which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
!
! In applying this licence, ECMWF does not waive the privileges and immunities
! granted to it by virtue of its status as an intergovernmental organisation
! nor does it submit to any jurisdiction.


MODULE CLOUDSC_GPU_OMP_SCC_MOD

integer, parameter :: JPRB = 4
integer, parameter :: JPIM = 4
integer, parameter :: NCLV = 32
REAL(KIND=JPRB) , parameter :: rtt = 0.01_JPRB
REAL(KIND=JPRB) , parameter :: r2es = 0.02_JPRB
REAL(KIND=JPRB) , parameter :: r3les = 0.03_JPRB
REAL(KIND=JPRB) , parameter :: r3ies = 0.04_JPRB
REAL(KIND=JPRB) , parameter :: r4les = 0.05_JPRB
REAL(KIND=JPRB) , parameter :: r4ies = 0.06_JPRB
REAL(KIND=JPRB) , parameter :: r5alvcp = 0.07_JPRB
REAL(KIND=JPRB) , parameter :: r5alscp = 0.08_JPRB
REAL(KIND=JPRB) , parameter :: r5les = 0.09_JPRB
REAL(KIND=JPRB) , parameter :: r5ies = 0.10_JPRB
REAL(KIND=JPRB) , parameter :: rlvtt = 0.11_JPRB
REAL(KIND=JPRB) , parameter :: rlstt = 0.12_JPRB
REAL(KIND=JPRB) , parameter :: ralvdcp = 0.13_JPRB
REAL(KIND=JPRB) , parameter :: ralsdcp = 0.14_JPRB
REAL(KIND=JPRB) , parameter :: rtice = 0.15_JPRB
REAL(KIND=JPRB) , parameter :: rtwat = 0.16_JPRB
REAL(KIND=JPRB) , parameter :: rtwat_rtice_r = 0.17_JPRB
REAL(KIND=JPRB) , parameter :: rticecu = 0.18_JPRB
REAL(KIND=JPRB) , parameter :: rtwat_rticecu_r = 0.19_JPRB
REAL(KIND=JPRB) , parameter :: rkoop1 = 0.20_JPRB
REAL(KIND=JPRB) , parameter :: rkoop2 = 0.21_JPRB
REAL(KIND=JPRB) , parameter :: rg = 0.22_JPRB
REAL(KIND=JPRB) , parameter :: rcpd = 0.23_JPRB
REAL(KIND=JPRB) , parameter :: rd = 0.24_JPRB
REAL(KIND=JPRB) , parameter :: rlmlt = 0.25_JPRB

integer, parameter :: ncldqv = 32
integer, parameter :: ncldql = 32
integer, parameter :: ncldqr = 32
integer, parameter :: ncldqi = 32
integer, parameter :: ncldqs = 32

REAL(KIND=JPRB) , parameter :: retv = 0.7_JPRB
REAL(KIND=JPRB) , parameter :: ralfdcp = 0.7_JPRB
REAL(KIND=JPRB) , parameter :: rv = 0.7_JPRB

TYPE :: TECLDP
REAL(KIND=JPRB) :: RAMID
REAL(KIND=JPRB) :: RCLDIFF
REAL(KIND=JPRB) :: RCLDIFF_CONVI
REAL(KIND=JPRB) :: RCLCRIT
REAL(KIND=JPRB) :: RCLCRIT_SEA
REAL(KIND=JPRB) :: RCLCRIT_LAND
REAL(KIND=JPRB) :: RKCONV
REAL(KIND=JPRB) :: RPRC1
REAL(KIND=JPRB) :: RPRC2
REAL(KIND=JPRB) :: RCLDMAX
REAL(KIND=JPRB) :: RPECONS
REAL(KIND=JPRB) :: RVRFACTOR
REAL(KIND=JPRB) :: RPRECRHMAX
REAL(KIND=JPRB) :: RTAUMEL
REAL(KIND=JPRB) :: RAMIN
REAL(KIND=JPRB) :: RLMIN
REAL(KIND=JPRB) :: RKOOPTAU
REAL(KIND=JPRB) :: RCLDTOPP
REAL(KIND=JPRB) :: RLCRITSNOW
REAL(KIND=JPRB) :: RSNOWLIN1
REAL(KIND=JPRB) :: RSNOWLIN2
REAL(KIND=JPRB) :: RICEHI1
REAL(KIND=JPRB) :: RICEHI2
REAL(KIND=JPRB) :: RICEINIT
REAL(KIND=JPRB) :: RVICE
REAL(KIND=JPRB) :: RVRAIN
REAL(KIND=JPRB) :: RVSNOW
REAL(KIND=JPRB) :: RTHOMO
REAL(KIND=JPRB) :: RCOVPMIN
REAL(KIND=JPRB) :: RCCN
REAL(KIND=JPRB) :: RNICE
REAL(KIND=JPRB) :: RCCNOM
REAL(KIND=JPRB) :: RCCNSS
REAL(KIND=JPRB) :: RCCNSU
REAL(KIND=JPRB) :: RCLDTOPCF
REAL(KIND=JPRB) :: RDEPLIQREFRATE
REAL(KIND=JPRB) :: RDEPLIQREFDEPTH
!--------------------------------------------------------
! Autoconversion/accretion (Khairoutdinov and Kogan 2000)
!--------------------------------------------------------
REAL(KIND=JPRB) :: RCL_KKAac
REAL(KIND=JPRB) :: RCL_KKBac
REAL(KIND=JPRB) :: RCL_KKAau
REAL(KIND=JPRB) :: RCL_KKBauq
REAL(KIND=JPRB) :: RCL_KKBaun
REAL(KIND=JPRB) :: RCL_KK_cloud_num_sea
REAL(KIND=JPRB) :: RCL_KK_cloud_num_land
!--------------------------------------------------------
! Ice
!--------------------------------------------------------
REAL(KIND=JPRB) :: RCL_AI
REAL(KIND=JPRB) :: RCL_BI
REAL(KIND=JPRB) :: RCL_CI
REAL(KIND=JPRB) :: RCL_DI
REAL(KIND=JPRB) :: RCL_X1I
REAL(KIND=JPRB) :: RCL_X2I
REAL(KIND=JPRB) :: RCL_X3I
REAL(KIND=JPRB) :: RCL_X4I
REAL(KIND=JPRB) :: RCL_CONST1I
REAL(KIND=JPRB) :: RCL_CONST2I
REAL(KIND=JPRB) :: RCL_CONST3I
REAL(KIND=JPRB) :: RCL_CONST4I
REAL(KIND=JPRB) :: RCL_CONST5I
REAL(KIND=JPRB) :: RCL_CONST6I
REAL(KIND=JPRB) :: RCL_APB1
REAL(KIND=JPRB) :: RCL_APB2
REAL(KIND=JPRB) :: RCL_APB3
!--------------------------------------------------------
! Snow
!--------------------------------------------------------
REAL(KIND=JPRB) :: RCL_AS
REAL(KIND=JPRB) :: RCL_BS
REAL(KIND=JPRB) :: RCL_CS
REAL(KIND=JPRB) :: RCL_DS
REAL(KIND=JPRB) :: RCL_X1S
REAL(KIND=JPRB) :: RCL_X2S
REAL(KIND=JPRB) :: RCL_X3S
REAL(KIND=JPRB) :: RCL_X4S
REAL(KIND=JPRB) :: RCL_CONST1S
REAL(KIND=JPRB) :: RCL_CONST2S
REAL(KIND=JPRB) :: RCL_CONST3S
REAL(KIND=JPRB) :: RCL_CONST4S
REAL(KIND=JPRB) :: RCL_CONST5S
REAL(KIND=JPRB) :: RCL_CONST6S
REAL(KIND=JPRB) :: RCL_CONST7S
REAL(KIND=JPRB) :: RCL_CONST8S
!--------------------------------------------------------
! Rain
!--------------------------------------------------------
REAL(KIND=JPRB) :: RDENSWAT
REAL(KIND=JPRB) :: RDENSREF
REAL(KIND=JPRB) :: RCL_AR
REAL(KIND=JPRB) :: RCL_BR
REAL(KIND=JPRB) :: RCL_CR
REAL(KIND=JPRB) :: RCL_DR
REAL(KIND=JPRB) :: RCL_X1R
REAL(KIND=JPRB) :: RCL_X2R
REAL(KIND=JPRB) :: RCL_X4R
REAL(KIND=JPRB) :: RCL_KA273
REAL(KIND=JPRB) :: RCL_CDENOM1
REAL(KIND=JPRB) :: RCL_CDENOM2
REAL(KIND=JPRB) :: RCL_CDENOM3
REAL(KIND=JPRB) :: RCL_SCHMIDT
REAL(KIND=JPRB) :: RCL_DYNVISC
REAL(KIND=JPRB) :: RCL_CONST1R
REAL(KIND=JPRB) :: RCL_CONST2R
REAL(KIND=JPRB) :: RCL_CONST3R
REAL(KIND=JPRB) :: RCL_CONST4R
REAL(KIND=JPRB) :: RCL_FAC1
REAL(KIND=JPRB) :: RCL_FAC2
! Rain freezing
REAL(KIND=JPRB) :: RCL_CONST5R
REAL(KIND=JPRB) :: RCL_CONST6R
REAL(KIND=JPRB) :: RCL_FZRAB
REAL(KIND=JPRB) :: RCL_FZRBB

LOGICAL :: LCLDEXTRA, LCLDBUDGET

INTEGER(KIND=JPIM) :: NSSOPT
INTEGER(KIND=JPIM) :: NCLDTOP
INTEGER(KIND=JPIM) :: NAECLBC, NAECLDU, NAECLOM, NAECLSS, NAECLSU
INTEGER(KIND=JPIM) :: NCLDDIAG

! aerosols
INTEGER(KIND=JPIM) :: NAERCLD
LOGICAL :: LAERLIQAUTOLSP
LOGICAL :: LAERLIQAUTOCP
LOGICAL :: LAERLIQAUTOCPB
LOGICAL :: LAERLIQCOLL
LOGICAL :: LAERICESED
LOGICAL :: LAERICEAUTO

! variance arrays
REAL(KIND=JPRB) :: NSHAPEP
REAL(KIND=JPRB) :: NSHAPEQ
INTEGER(KIND=JPIM) :: NBETA
REAL(KIND=JPRB) :: RBETA(0:100)
REAL(KIND=JPRB) :: RBETAP1(0:100)

END TYPE TECLDP
TYPE(TECLDP), ALLOCATABLE :: YRECLDP

!$omp declare target(CLOUDSC_SCC)

CONTAINS
  SUBROUTINE CLOUDSC_SCC (JL, KIDIA, KFDIA, KLON, KLEV, PTSPHY, PT, PQ, TENDENCY_TMP_T, TENDENCY_TMP_Q, TENDENCY_TMP_A,  &
  & TENDENCY_TMP_CLD, TENDENCY_LOC_T, TENDENCY_LOC_Q, TENDENCY_LOC_A, TENDENCY_LOC_CLD, PVFA, PVFL, PVFI, PDYNA, PDYNL, PDYNI,  &
  & PHRSW, PHRLW, PVERVEL, PAP, PAPH, PLSM, LDCUM, KTYPE, PLU, PLUDE, PSNDE, PMFU, PMFD, PA, PCLV, PSUPSAT, PLCRIT_AER,  &
  & PICRIT_AER, PRE_ICE, PCCN, PNICE, PCOVPTOT, PRAINFRAC_TOPRFZ, PFSQLF, PFSQIF, PFCQNNG, PFCQLNG, PFSQRF, PFSQSF, PFCQRNG,  &
  & PFCQSNG, PFSQLTUR, PFSQITUR, PFPLSL, PFPLSN, PFHPSL, PFHPSN, &
  & ZFOEALFA, ZTP1, ZLI, ZA, ZAORIG, ZLIQFRAC, ZICEFRAC, ZQX, ZQX0, ZPFPLSX, &
  & ZLNEG, ZQXN2D, ZQSMIX, ZQSLIQ, ZQSICE, ZFOEEWMT, ZFOEEW, ZFOEELIQT, &
  & YRECLDP)
    !---input
    !---prognostic fields
    !-- arrays for aerosol-cloud interactions
    !!! & PQAER,    KAER, &
    !---diagnostic output
    !---resulting fluxes

    !===============================================================================
    !**** *CLOUDSC* -  ROUTINE FOR PARAMATERIZATION OF CLOUD PROCESSES
    !                  FOR PROGNOSTIC CLOUD SCHEME
    !!
    !     M.Tiedtke, C.Jakob, A.Tompkins, R.Forbes     (E.C.M.W.F.)
    !!
    !     PURPOSE
    !     -------
    !          THIS ROUTINE UPDATES THE CONV/STRAT CLOUD FIELDS.
    !          THE FOLLOWING PROCESSES ARE CONSIDERED:
    !        - Detrainment of cloud water from convective updrafts
    !        - Evaporation/condensation of cloud water in connection
    !           with heating/cooling such as by subsidence/ascent
    !        - Erosion of clouds by turbulent mixing of cloud air
    !           with unsaturated environmental air
    !        - Deposition onto ice when liquid water present (Bergeron-Findeison)
    !        - Conversion of cloud water into rain (collision-coalescence)
    !        - Conversion of cloud ice to snow (aggregation)
    !        - Sedimentation of rain, snow and ice
    !        - Evaporation of rain and snow
    !        - Melting of snow and ice
    !        - Freezing of liquid and rain
    !        Note: Turbulent transports of s,q,u,v at cloud tops due to
    !           buoyancy fluxes and lw radiative cooling are treated in
    !           the VDF scheme
    !!
    !     INTERFACE.
    !     ----------
    !          *CLOUDSC* IS CALLED FROM *CALLPAR*
    !     THE ROUTINE TAKES ITS INPUT FROM THE LONG-TERM STORAGE:
    !     T,Q,L,PHI AND DETRAINMENT OF CLOUD WATER FROM THE
    !     CONVECTIVE CLOUDS (MASSFLUX CONVECTION SCHEME), BOUNDARY
    !     LAYER TURBULENT FLUXES OF HEAT AND MOISTURE, RADIATIVE FLUXES,
    !     OMEGA.
    !     IT RETURNS ITS OUTPUT TO:
    !      1.MODIFIED TENDENCIES OF MODEL VARIABLES T AND Q
    !        AS WELL AS CLOUD VARIABLES L AND C
    !      2.GENERATES PRECIPITATION FLUXES FROM STRATIFORM CLOUDS
    !!
    !     EXTERNALS.
    !     ----------
    !          NONE
    !!
    !     MODIFICATIONS.
    !     -------------
    !      M. TIEDTKE    E.C.M.W.F.     8/1988, 2/1990
    !     CH. JAKOB      E.C.M.W.F.     2/1994 IMPLEMENTATION INTO IFS
    !     A.TOMPKINS     E.C.M.W.F.     2002   NEW NUMERICS
    !        01-05-22 : D.Salmond   Safety modifications
    !        02-05-29 : D.Salmond   Optimisation
    !        03-01-13 : J.Hague     MASS Vector Functions  J.Hague
    !        03-10-01 : M.Hamrud    Cleaning
    !        04-12-14 : A.Tompkins  New implicit solver and physics changes
    !        04-12-03 : A.Tompkins & M.Ko"hler  moist PBL
    !     G.Mozdzynski  09-Jan-2006  EXP security fix
    !        19-01-09 : P.Bechtold  Changed increased RCLDIFF value for KTYPE=2
    !        07-07-10 : A.Tompkins/R.Forbes  4-Phase flexible microphysics
    !        01-03-11 : R.Forbes    Mixed phase changes and tidy up
    !        01-10-11 : R.Forbes    Melt ice to rain, allow rain to freeze
    !        01-10-11 : R.Forbes    Limit supersat to avoid excessive values
    !        31-10-11 : M.Ahlgrimm  Add rain, snow and PEXTRA to DDH output
    !        17-02-12 : F.Vana      Simplified/optimized LU factorization
    !        18-05-12 : F.Vana      Cleaning + better support of sequential physics
    !        N.Semane+P.Bechtold     04-10-2012 Add RVRFACTOR factor for small planet
    !        01-02-13 : R.Forbes    New params of autoconv/acc,rain evap,snow riming
    !        15-03-13 : F. Vana     New dataflow + more tendencies from the first call
    !        K. Yessad (July 2014): Move some variables.
    !        F. Vana  05-Mar-2015  Support for single precision
    !        15-01-15 : R.Forbes    Added new options for snow evap & ice deposition
    !        10-01-15 : R.Forbes    New physics for rain freezing
    !        23-10-14 : P. Bechtold remove zeroing of convection arrays
    !
    !     SWITCHES.
    !     --------
    !!
    !     MODEL PARAMETERS
    !     ----------------
    !     RCLDIFF:    PARAMETER FOR EROSION OF CLOUDS
    !     RCLCRIT_SEA:  THRESHOLD VALUE FOR RAIN AUTOCONVERSION OVER SEA
    !     RCLCRIT_LAND: THRESHOLD VALUE FOR RAIN AUTOCONVERSION OVER LAND
    !     RLCRITSNOW: THRESHOLD VALUE FOR SNOW AUTOCONVERSION
    !     RKCONV:     PARAMETER FOR AUTOCONVERSION OF CLOUDS (KESSLER)
    !     RCLDMAX:    MAXIMUM POSSIBLE CLW CONTENT (MASON,1971)
    !!
    !     REFERENCES.
    !     ----------
    !     TIEDTKE MWR 1993
    !     JAKOB PhD 2000
    !     GREGORY ET AL. QJRMS 2000
    !     TOMPKINS ET AL. QJRMS 2007
    !!
    !===============================================================================

!     USE PARKIND1, ONLY: JPIM, JPRB
!     USE YOMCST, ONLY: RG, RD, RCPD, RETV, RLVTT, RLSTT, RLMLT, RTT, RV
!     USE YOETHF, ONLY: R2ES, R3LES, R3IES, R4LES, R4IES, R5LES, R5IES, R5ALVCP, R5ALSCP, RALVDCP, RALSDCP, RALFDCP, RTWAT, RTICE,  &
!     & RTICECU, RTWAT_RTICE_R, RTWAT_RTICECU_R, RKOOP1, RKOOP2
!     USE YOECLDP, ONLY: TECLDP, NCLDQV, NCLDQL, NCLDQR, NCLDQI, NCLDQS, NCLV


    IMPLICIT NONE

    !-------------------------------------------------------------------------------
    !                 Declare input/output arguments
    !-------------------------------------------------------------------------------

    ! PLCRIT_AER : critical liquid mmr for rain autoconversion process
    ! PICRIT_AER : critical liquid mmr for snow autoconversion process
    ! PRE_LIQ : liq Re
    ! PRE_ICE : ice Re
    ! PCCN    : liquid cloud condensation nuclei
    ! PNICE   : ice number concentration (cf. CCN)

    REAL(KIND=JPRB), INTENT(IN) :: PLCRIT_AER(KLON, KLEV)
    REAL(KIND=JPRB), INTENT(IN) :: PICRIT_AER(KLON, KLEV)
    REAL(KIND=JPRB), INTENT(IN) :: PRE_ICE(KLON, KLEV)
    REAL(KIND=JPRB), INTENT(IN) :: PCCN(KLON, KLEV)    ! liquid cloud condensation nuclei
    REAL(KIND=JPRB), INTENT(IN) :: PNICE(KLON, KLEV)
    ! ice number concentration (cf. CCN)

    INTEGER(KIND=JPIM), INTENT(IN) :: JL
    INTEGER(KIND=JPIM), INTENT(IN) :: KLON    ! Number of grid points
    INTEGER(KIND=JPIM), INTENT(IN) :: KLEV    ! Number of levels
    INTEGER(KIND=JPIM), INTENT(IN) :: KIDIA
    INTEGER(KIND=JPIM), INTENT(IN) :: KFDIA
    REAL(KIND=JPRB), INTENT(IN) :: PTSPHY    ! Physics timestep
    REAL(KIND=JPRB), INTENT(IN) :: PT(KLON, KLEV)    ! T at start of callpar
    REAL(KIND=JPRB), INTENT(IN) :: PQ(KLON, KLEV)    ! Q at start of callpar
    REAL(KIND=JPRB), INTENT(IN) :: TENDENCY_TMP_T(KLON, KLEV)
    REAL(KIND=JPRB), INTENT(IN) :: TENDENCY_TMP_Q(KLON, KLEV)
    REAL(KIND=JPRB), INTENT(IN) :: TENDENCY_TMP_A(KLON, KLEV)
    REAL(KIND=JPRB), INTENT(IN) :: TENDENCY_TMP_CLD(KLON, KLEV, NCLV)
    REAL(KIND=JPRB), INTENT(INOUT) :: TENDENCY_LOC_T(KLON, KLEV)
    REAL(KIND=JPRB), INTENT(INOUT) :: TENDENCY_LOC_Q(KLON, KLEV)
    REAL(KIND=JPRB), INTENT(INOUT) :: TENDENCY_LOC_A(KLON, KLEV)
    REAL(KIND=JPRB), INTENT(INOUT) :: TENDENCY_LOC_CLD(KLON, KLEV, NCLV)
    REAL(KIND=JPRB), INTENT(IN) :: PVFA(KLON, KLEV)    ! CC from VDF scheme
    REAL(KIND=JPRB), INTENT(IN) :: PVFL(KLON, KLEV)    ! Liq from VDF scheme
    REAL(KIND=JPRB), INTENT(IN) :: PVFI(KLON, KLEV)    ! Ice from VDF scheme
    REAL(KIND=JPRB), INTENT(IN) :: PDYNA(KLON, KLEV)    ! CC from Dynamics
    REAL(KIND=JPRB), INTENT(IN) :: PDYNL(KLON, KLEV)    ! Liq from Dynamics
    REAL(KIND=JPRB), INTENT(IN) :: PDYNI(KLON, KLEV)    ! Liq from Dynamics
    REAL(KIND=JPRB), INTENT(IN) :: PHRSW(KLON, KLEV)    ! Short-wave heating rate
    REAL(KIND=JPRB), INTENT(IN) :: PHRLW(KLON, KLEV)    ! Long-wave heating rate
    REAL(KIND=JPRB), INTENT(IN) :: PVERVEL(KLON, KLEV)    !Vertical velocity
    REAL(KIND=JPRB), INTENT(IN) :: PAP(KLON, KLEV)    ! Pressure on full levels
    REAL(KIND=JPRB), INTENT(IN) :: PAPH(KLON, KLEV + 1)    ! Pressure on half levels
    REAL(KIND=JPRB), INTENT(IN) :: PLSM(KLON)    ! Land fraction (0-1)
    LOGICAL, INTENT(IN) :: LDCUM(KLON)    ! Convection active
    INTEGER(KIND=JPIM), INTENT(IN) :: KTYPE(KLON)    ! Convection type 0,1,2
    REAL(KIND=JPRB), INTENT(IN) :: PLU(KLON, KLEV)    ! Conv. condensate
    REAL(KIND=JPRB), INTENT(INOUT) :: PLUDE(KLON, KLEV)    ! Conv. detrained water
    REAL(KIND=JPRB), INTENT(IN) :: PSNDE(KLON, KLEV)    ! Conv. detrained snow
    REAL(KIND=JPRB), INTENT(IN) :: PMFU(KLON, KLEV)    ! Conv. mass flux up
    REAL(KIND=JPRB), INTENT(IN) :: PMFD(KLON, KLEV)    ! Conv. mass flux down
    REAL(KIND=JPRB), INTENT(IN) :: PA(KLON, KLEV)
    ! Original Cloud fraction (t)

    REAL(KIND=JPRB), INTENT(IN) :: PCLV(KLON, KLEV, NCLV)

    ! Supersat clipped at previous time level in SLTEND
    REAL(KIND=JPRB), INTENT(IN) :: PSUPSAT(KLON, KLEV)
    REAL(KIND=JPRB), INTENT(OUT) :: PCOVPTOT(KLON, KLEV)    ! Precip fraction
    REAL(KIND=JPRB), INTENT(OUT) :: PRAINFRAC_TOPRFZ(KLON)
    ! Flux diagnostics for DDH budget
    REAL(KIND=JPRB), INTENT(OUT) :: PFSQLF(KLON, KLEV + 1)    ! Flux of liquid
    REAL(KIND=JPRB), INTENT(OUT) :: PFSQIF(KLON, KLEV + 1)    ! Flux of ice
    REAL(KIND=JPRB), INTENT(OUT) :: PFCQLNG(KLON, KLEV + 1)    ! -ve corr for liq
    REAL(KIND=JPRB), INTENT(OUT) :: PFCQNNG(KLON, KLEV + 1)    ! -ve corr for ice
    REAL(KIND=JPRB), INTENT(OUT) :: PFSQRF(KLON, KLEV + 1)    ! Flux diagnostics
    REAL(KIND=JPRB), INTENT(OUT) :: PFSQSF(KLON, KLEV + 1)    !    for DDH, generic
    REAL(KIND=JPRB), INTENT(OUT) :: PFCQRNG(KLON, KLEV + 1)    ! rain
    REAL(KIND=JPRB), INTENT(OUT) :: PFCQSNG(KLON, KLEV + 1)    ! snow
    REAL(KIND=JPRB), INTENT(OUT) :: PFSQLTUR(KLON, KLEV + 1)    ! liquid flux due to VDF
    REAL(KIND=JPRB), INTENT(OUT) :: PFSQITUR(KLON, KLEV + 1)    ! ice flux due to VDF
    REAL(KIND=JPRB), INTENT(OUT) :: PFPLSL(KLON, KLEV + 1)    ! liq+rain sedim flux
    REAL(KIND=JPRB), INTENT(OUT) :: PFPLSN(KLON, KLEV + 1)    ! ice+snow sedim flux
    REAL(KIND=JPRB), INTENT(OUT) :: PFHPSL(KLON, KLEV + 1)    ! Enthalpy flux for liq
    REAL(KIND=JPRB), INTENT(OUT) :: PFHPSN(KLON, KLEV + 1)
    ! Enthalp flux for ice

    TYPE(tecldp), INTENT(INOUT) :: YRECLDP

    !-------------------------------------------------------------------------------
    !                       Declare local variables
    !-------------------------------------------------------------------------------

    REAL(KIND=JPRB) :: ZLCOND1, ZLCOND2, ZLEVAP, ZLEROS, ZLEVAPL, ZLEVAPI, ZRAINAUT, ZSNOWAUT, ZLIQCLD, ZICECLD
    !  condensation and evaporation terms
    ! autoconversion terms
    REAL(KIND=JPRB) :: ZFOKOOP
    !REAL(KIND=JPRB) :: ZFOEALFA(KLON, KLEV + 1)
    REAL(KIND=JPRB) :: ZICENUCLEI
    ! number concentration of ice nuclei

    REAL(KIND=JPRB) :: ZLICLD
    REAL(KIND=JPRB) :: ZACOND
    REAL(KIND=JPRB) :: ZAEROS
    REAL(KIND=JPRB) :: ZLFINALSUM
    REAL(KIND=JPRB) :: ZDQS
    REAL(KIND=JPRB) :: ZTOLD
    REAL(KIND=JPRB) :: ZQOLD
    REAL(KIND=JPRB) :: ZDTGDP
    REAL(KIND=JPRB) :: ZRDTGDP
    REAL(KIND=JPRB) :: ZTRPAUS
    REAL(KIND=JPRB) :: ZCOVPCLR
    REAL(KIND=JPRB) :: ZPRECLR
    REAL(KIND=JPRB) :: ZCOVPTOT
    REAL(KIND=JPRB) :: ZCOVPMAX
    REAL(KIND=JPRB) :: ZQPRETOT
    REAL(KIND=JPRB) :: ZDPEVAP
    REAL(KIND=JPRB) :: ZDTFORC
    REAL(KIND=JPRB) :: ZDTDIAB
    !REAL(KIND=JPRB) :: ZTP1(KLON, KLEV)
    REAL(KIND=JPRB) :: ZLDEFR
    REAL(KIND=JPRB) :: ZLDIFDT
    REAL(KIND=JPRB) :: ZDTGDPF
    !REAL(KIND=JPRB) :: ZLCUST(NCLV)
    REAL(KIND=JPRB) :: ZACUST
    REAL(KIND=JPRB) :: ZMF

    REAL(KIND=JPRB) :: ZRHO
    REAL(KIND=JPRB) :: ZTMP1, ZTMP2, ZTMP3
    REAL(KIND=JPRB) :: ZTMP4, ZTMP5, ZTMP6, ZTMP7
    REAL(KIND=JPRB) :: ZALFAWM

    ! Accumulators of A,B,and C factors for cloud equations
    REAL(KIND=JPRB) :: ZSOLAB    ! -ve implicit CC
    REAL(KIND=JPRB) :: ZSOLAC    ! linear CC
    REAL(KIND=JPRB) :: ZANEW
    REAL(KIND=JPRB) :: ZANEWM1

    REAL(KIND=JPRB) :: ZGDP

    !---for flux calculation
    REAL(KIND=JPRB) :: ZDA
    !REAL(KIND=JPRB) :: ZLI(KLON, KLEV), ZA(KLON, KLEV)
    !REAL(KIND=JPRB) :: ZAORIG(KLON, KLEV)
    ! start of scheme value for CC

    LOGICAL :: LLFLAG
    LOGICAL :: LLO1

    INTEGER(KIND=JPIM) :: ICALL, IK, JK, JM, JN, JO, JLEN, IS

    REAL(KIND=JPRB) :: ZDP, ZPAPHD

    REAL(KIND=JPRB) :: ZALFA
    ! & ZALFACU, ZALFALS
    REAL(KIND=JPRB) :: ZALFAW
    REAL(KIND=JPRB) :: ZBETA, ZBETA1
    !REAL(KIND=JPRB) :: ZBOTT
    REAL(KIND=JPRB) :: ZCFPR
    REAL(KIND=JPRB) :: ZCOR
    REAL(KIND=JPRB) :: ZCDMAX
    REAL(KIND=JPRB) :: ZMIN
    REAL(KIND=JPRB) :: ZLCONDLIM
    REAL(KIND=JPRB) :: ZDENOM
    REAL(KIND=JPRB) :: ZDPMXDT
    REAL(KIND=JPRB) :: ZDPR
    REAL(KIND=JPRB) :: ZDTDP
    REAL(KIND=JPRB) :: ZE
    REAL(KIND=JPRB) :: ZEPSEC
    REAL(KIND=JPRB) :: ZFAC, ZFACI, ZFACW
    REAL(KIND=JPRB) :: ZGDCP
    REAL(KIND=JPRB) :: ZINEW
    REAL(KIND=JPRB) :: ZLCRIT
    REAL(KIND=JPRB) :: ZMFDN
    REAL(KIND=JPRB) :: ZPRECIP
    REAL(KIND=JPRB) :: ZQE
    REAL(KIND=JPRB) :: ZQSAT, ZQTMST, ZRDCP
    REAL(KIND=JPRB) :: ZRHC, ZSIG, ZSIGK
    REAL(KIND=JPRB) :: ZWTOT
    REAL(KIND=JPRB) :: ZZCO, ZZDL, ZZRH, ZZZDT, ZQADJ
    REAL(KIND=JPRB) :: ZQNEW, ZTNEW
    REAL(KIND=JPRB) :: ZRG_R, ZGDPH_R, ZCONS1, ZCOND, ZCONS1A
    REAL(KIND=JPRB) :: ZLFINAL
    REAL(KIND=JPRB) :: ZMELT
    REAL(KIND=JPRB) :: ZEVAP
    REAL(KIND=JPRB) :: ZFRZ
    REAL(KIND=JPRB) :: ZVPLIQ, ZVPICE
    REAL(KIND=JPRB) :: ZADD, ZBDD, ZCVDS, ZICE0, ZDEPOS
    REAL(KIND=JPRB) :: ZSUPSAT
    REAL(KIND=JPRB) :: ZFALL
    REAL(KIND=JPRB) :: ZRE_ICE
    REAL(KIND=JPRB) :: ZRLDCP
    REAL(KIND=JPRB) :: ZQP1ENV

    !----------------------------
    ! Arrays for new microphysics
    !----------------------------
    !INTEGER(KIND=JPIM) :: IPHASE(NCLV)
    ! marker for water phase of each species
    ! 0=vapour, 1=liquid, 2=ice

    !INTEGER(KIND=JPIM) :: IMELT(NCLV)
    ! marks melting linkage for ice categories
    ! ice->liquid, snow->rain

    !LOGICAL :: LLFALL(NCLV)
    ! marks falling species
    ! LLFALL=0, cloud cover must > 0 for zqx > 0
    ! LLFALL=1, no cloud needed, zqx can evaporate

    !LOGICAL :: LLINDEX1(NCLV)    ! index variable
    !LOGICAL :: LLINDEX3(NCLV, NCLV)    ! index variable
    REAL(KIND=JPRB) :: ZMAX
    REAL(KIND=JPRB) :: ZRAT
    !INTEGER(KIND=JPIM) :: IORDER(NCLV)
    ! array for sorting explicit terms

    !REAL(KIND=JPRB) :: ZLIQFRAC(KLON, KLEV)    ! cloud liquid water fraction: ql/(ql+qi)
    !REAL(KIND=JPRB) :: ZICEFRAC(KLON, KLEV)    ! cloud ice water fraction: qi/(ql+qi)
    !REAL(KIND=JPRB) :: ZQX(KLON, KLEV, NCLV)    ! water variables
    !REAL(KIND=JPRB) :: ZQX0(KLON, KLEV, NCLV)    ! water variables at start of scheme
    !REAL(KIND=JPRB) :: ZQXN(NCLV)    ! new values for zqx at time+1
    !REAL(KIND=JPRB) :: ZQXFG(NCLV)    ! first guess values including precip
    !REAL(KIND=JPRB) :: ZQXNM1(NCLV)    ! new values for zqx at time+1 at level above
    !REAL(KIND=JPRB) :: ZFLUXQ(NCLV)
    ! fluxes convergence of species (needed?)
    ! Keep the following for possible future total water variance scheme?
    !REAL(KIND=JPRB) :: ZTL(KLON,KLEV)       ! liquid water temperature
    !REAL(KIND=JPRB) :: ZABETA(KLON,KLEV)    ! cloud fraction
    !REAL(KIND=JPRB) :: ZVAR(KLON,KLEV)      ! temporary variance
    !REAL(KIND=JPRB) :: ZQTMIN(KLON,KLEV)
    !REAL(KIND=JPRB) :: ZQTMAX(KLON,KLEV)

    !REAL(KIND=JPRB) :: ZPFPLSX(KLON, KLEV + 1, NCLV)    ! generalized precipitation flux
    !REAL(KIND=JPRB) :: ZLNEG(KLON, KLEV, NCLV)    ! for negative correction diagnostics
    REAL(KIND=JPRB) :: ZMELTMAX
    REAL(KIND=JPRB) :: ZFRZMAX
    REAL(KIND=JPRB) :: ZICETOT

    !REAL(KIND=JPRB) :: ZQXN2D(KLON, KLEV, NCLV)
    ! water variables store

    !REAL(KIND=JPRB) :: ZQSMIX(KLON, KLEV)
    ! diagnostic mixed phase saturation
    !REAL(KIND=JPRB) :: ZQSBIN(KLON,KLEV) ! binary switched ice/liq saturation
    !REAL(KIND=JPRB) :: ZQSLIQ(KLON, KLEV)    ! liquid water saturation
    !REAL(KIND=JPRB) :: ZQSICE(KLON, KLEV)
    ! ice water saturation

    !REAL(KIND=JPRB) :: ZRHM(KLON,KLEV) ! diagnostic mixed phase RH
    !REAL(KIND=JPRB) :: ZRHL(KLON,KLEV) ! RH wrt liq
    !REAL(KIND=JPRB) :: ZRHI(KLON,KLEV) ! RH wrt ice

    !REAL(KIND=JPRB) :: ZFOEEWMT(KLON, KLEV)
    !REAL(KIND=JPRB) :: ZFOEEW(KLON, KLEV)
    !REAL(KIND=JPRB) :: ZFOEELIQT(KLON, KLEV)
    !REAL(KIND=JPRB) :: ZFOEEICET(KLON,KLEV)

    REAL(KIND=JPRB) :: ZDQSLIQDT, ZDQSICEDT, ZDQSMIXDT
    REAL(KIND=JPRB) :: ZCORQSLIQ
    REAL(KIND=JPRB) :: ZCORQSICE
    !REAL(KIND=JPRB) :: ZCORQSBIN(KLON)
    REAL(KIND=JPRB) :: ZCORQSMIX
    REAL(KIND=JPRB) :: ZEVAPLIMLIQ, ZEVAPLIMICE, ZEVAPLIMMIX

    !-------------------------------------------------------
    ! SOURCE/SINK array for implicit and explicit terms
    !-------------------------------------------------------
    ! a POSITIVE value entered into the arrays is a...
    !            Source of this variable
    !            |
    !            |   Sink of this variable
    !            |   |
    !            V   V
    ! ZSOLQA(JL,IQa,IQb)  = explicit terms
    ! ZSOLQB(JL,IQa,IQb)  = implicit terms
    ! Thus if ZSOLAB(JL,NCLDQL,IQV)=K where K>0 then this is
    ! a source of NCLDQL and a sink of IQV
    ! put 'magic' source terms such as PLUDE from
    ! detrainment into explicit source/sink array diagnognal
    ! ZSOLQA(NCLDQL,NCLDQL)= -PLUDE
    ! i.e. A positive value is a sink!????? weird...
    !-------------------------------------------------------

    !REAL(KIND=JPRB) :: ZSOLQA(NCLV, NCLV)    ! explicit sources and sinks
    !REAL(KIND=JPRB) :: ZSOLQB(NCLV, NCLV)
    ! implicit sources and sinks
    ! e.g. microphysical pathways between ice variables.
    !REAL(KIND=JPRB) :: ZQLHS(NCLV, NCLV)    ! n x n matrix storing the LHS of implicit solver
    !REAL(KIND=JPRB) :: ZVQX(NCLV)    ! fall speeds of three categories
    !REAL(KIND=JPRB) :: ZEXPLICIT, ZRATIO(NCLV), ZSINKSUM(NCLV)
    REAL(KIND=JPRB) :: ZEXPLICIT

    ! for sedimentation source/sink terms
    !REAL(KIND=JPRB) :: ZFALLSINK(NCLV)
    !REAL(KIND=JPRB) :: ZFALLSRCE(NCLV)

    ! for convection detrainment source and subsidence source/sink terms
    !REAL(KIND=JPRB) :: ZCONVSRCE(NCLV)
    !REAL(KIND=JPRB) :: ZCONVSINK(NCLV)

    ! for supersaturation source term from previous timestep
    !REAL(KIND=JPRB) :: ZPSUPSATSRCE(NCLV)

    ! Numerical fit to wet bulb temperature
    REAL(KIND=JPRB), PARAMETER :: ZTW1 = 1329.31_4
    REAL(KIND=JPRB), PARAMETER :: ZTW2 = 0.0074615_4
    REAL(KIND=JPRB), PARAMETER :: ZTW3 = 0.85E5_4
    REAL(KIND=JPRB), PARAMETER :: ZTW4 = 40.637_4
    REAL(KIND=JPRB), PARAMETER :: ZTW5 = 275.0_4

    REAL(KIND=JPRB) :: ZSUBSAT    ! Subsaturation for snow melting term
    REAL(KIND=JPRB) :: ZTDMTW0
    ! Diff between dry-bulb temperature and
    ! temperature when wet-bulb = 0degC

    ! Variables for deposition term
    REAL(KIND=JPRB) :: ZTCG    ! Temperature dependent function for ice PSD
    REAL(KIND=JPRB) :: ZFACX1I, ZFACX1S    ! PSD correction factor
    REAL(KIND=JPRB) :: ZAPLUSB, ZCORRFAC, ZCORRFAC2, ZPR02, ZTERM1, ZTERM2    ! for ice dep
    REAL(KIND=JPRB) :: ZCLDTOPDIST    ! Distance from cloud top
    REAL(KIND=JPRB) :: ZINFACTOR
    ! No. of ice nuclei factor for deposition

    ! Autoconversion/accretion/riming/evaporation
    INTEGER(KIND=JPIM) :: IWARMRAIN
    INTEGER(KIND=JPIM) :: IEVAPRAIN
    INTEGER(KIND=JPIM) :: IEVAPSNOW
    INTEGER(KIND=JPIM) :: IDEPICE
    REAL(KIND=JPRB) :: ZRAINACC
    REAL(KIND=JPRB) :: ZRAINCLD
    REAL(KIND=JPRB) :: ZSNOWRIME
    REAL(KIND=JPRB) :: ZSNOWCLD
    REAL(KIND=JPRB) :: ZESATLIQ
    REAL(KIND=JPRB) :: ZFALLCORR
    REAL(KIND=JPRB) :: ZLAMBDA
    REAL(KIND=JPRB) :: ZEVAP_DENOM
    REAL(KIND=JPRB) :: ZCORR2
    REAL(KIND=JPRB) :: ZKA
    REAL(KIND=JPRB) :: ZCONST
    REAL(KIND=JPRB) :: ZTEMP

    ! Rain freezing
    LOGICAL :: LLRAINLIQ
    ! True if majority of raindrops are liquid (no ice core)

    !----------------------------
    ! End: new microphysics
    !----------------------------

    !----------------------
    ! SCM budget statistics
    !----------------------
    REAL(KIND=JPRB) :: ZRAIN

    REAL(KIND=JPRB) :: ZHOOK_HANDLE
    REAL(KIND=JPRB) :: ZTMPL, ZTMPI, ZTMPA

    REAL(KIND=JPRB) :: ZMM, ZRR
    REAL(KIND=JPRB) :: ZRG

    REAL(KIND=JPRB) :: ZZSUM, ZZRATIO
    REAL(KIND=JPRB) :: ZEPSILON

    REAL(KIND=JPRB) :: ZCOND1, ZQP

    REAL(KIND=JPRB) :: PSUM_SOLQA

#include "fcttre.func.h"
#include "fccld.func.h"

    REAL(KIND=JPRB) :: ZLCUST(NCLV)
    INTEGER(KIND=JPIM) :: IPHASE(NCLV)
    INTEGER(KIND=JPIM) :: IMELT(NCLV)
    LOGICAL :: LLFALL(NCLV)
    LOGICAL :: LLINDEX1(NCLV)
    LOGICAL :: LLINDEX3(NCLV, NCLV) !
    INTEGER(KIND=JPIM) :: IORDER(NCLV)
    REAL(KIND=JPRB) :: ZQXN(NCLV)
    REAL(KIND=JPRB) :: ZQXFG(NCLV)
    REAL(KIND=JPRB) :: ZQXNM1(NCLV)
    REAL(KIND=JPRB) :: ZFLUXQ(NCLV)
    REAL(KIND=JPRB) :: ZSOLQA(NCLV, NCLV) !
    REAL(KIND=JPRB) :: ZSOLQB(NCLV, NCLV) !
    REAL(KIND=JPRB) :: ZQLHS(NCLV, NCLV) !
    REAL(KIND=JPRB) :: ZVQX(NCLV)
    REAL(KIND=JPRB) :: ZRATIO(NCLV)
    REAL(KIND=JPRB) :: ZSINKSUM(NCLV)
    REAL(KIND=JPRB) :: ZFALLSINK(NCLV)
    REAL(KIND=JPRB) :: ZFALLSRCE(NCLV)
    REAL(KIND=JPRB) :: ZCONVSRCE(NCLV)
    REAL(KIND=JPRB) :: ZCONVSINK(NCLV)
    REAL(KIND=JPRB) :: ZPSUPSATSRCE(NCLV)

    REAL(KIND=JPRB),intent(inout) :: ZFOEALFA(KLON, KLEV + 1)
    REAL(KIND=JPRB),intent(inout) :: ZTP1(KLON, KLEV)
    REAL(KIND=JPRB),intent(inout) :: ZLI(KLON, KLEV), ZA(KLON, KLEV)
    REAL(KIND=JPRB),intent(inout) :: ZAORIG(KLON, KLEV)
    REAL(KIND=JPRB),intent(inout) :: ZLIQFRAC(KLON, KLEV)
    REAL(KIND=JPRB),intent(inout) :: ZICEFRAC(KLON, KLEV)
    REAL(KIND=JPRB),intent(inout) :: ZQX(KLON, KLEV, KLON)
    REAL(KIND=JPRB),intent(inout) :: ZQX0(KLON, KLEV, KLON)
    REAL(KIND=JPRB),intent(inout) :: ZPFPLSX(KLON, KLEV + 1, KLON)
    REAL(KIND=JPRB),intent(inout) :: ZLNEG(KLON, KLEV, KLON)
    REAL(KIND=JPRB),intent(inout) :: ZQXN2D(KLON, KLEV, KLON)
    REAL(KIND=JPRB),intent(inout) :: ZQSMIX(KLON, KLEV)
    REAL(KIND=JPRB),intent(inout) :: ZQSLIQ(KLON, KLEV)
    REAL(KIND=JPRB),intent(inout) :: ZQSICE(KLON, KLEV)
    REAL(KIND=JPRB),intent(inout) :: ZFOEEWMT(KLON, KLEV)
    REAL(KIND=JPRB),intent(inout) :: ZFOEEW(KLON, KLEV)
    REAL(KIND=JPRB),intent(inout) :: ZFOEELIQT(KLON, KLEV)

    !DO JL=KIDIA,KFDIA


      !===============================================================================
      !IF (LHOOK) CALL DR_HOOK('CLOUDSC',0,ZHOOK_HANDLE)

      !===============================================================================
      !  0.0     Beginning of timestep book-keeping
      !----------------------------------------------------------------------


      !######################################################################
      !             0.  *** SET UP CONSTANTS ***
      !######################################################################

      ZEPSILON = 100._JPRB*EPSILON(ZEPSILON)

      ! ---------------------------------------------------------------------
      ! Set version of warm-rain autoconversion/accretion
      ! IWARMRAIN = 1 ! Sundquist
      ! IWARMRAIN = 2 ! Khairoutdinov and Kogan (2000)
      ! ---------------------------------------------------------------------
      IWARMRAIN = 2
      ! ---------------------------------------------------------------------
      ! Set version of rain evaporation
      ! IEVAPRAIN = 1 ! Sundquist
      ! IEVAPRAIN = 2 ! Abel and Boutle (2013)
      ! ---------------------------------------------------------------------
      IEVAPRAIN = 2
      ! ---------------------------------------------------------------------
      ! Set version of snow evaporation
      ! IEVAPSNOW = 1 ! Sundquist
      ! IEVAPSNOW = 2 ! New
      ! ---------------------------------------------------------------------
      IEVAPSNOW = 1
      ! ---------------------------------------------------------------------
      ! Set version of ice deposition
      ! IDEPICE = 1 ! Rotstayn (2001)
      ! IDEPICE = 2 ! New
      ! ---------------------------------------------------------------------
      IDEPICE = 1

      ! ---------------------
      ! Some simple constants
      ! ---------------------
      ZQTMST = 1.0_JPRB / PTSPHY
      ZGDCP = RG / RCPD
      ZRDCP = RD / RCPD
      ZCONS1A = RCPD / ((RLMLT*RG*YRECLDP%RTAUMEL))
      ZEPSEC = 1.E-14_JPRB
      ZRG_R = 1.0_JPRB / RG
      ZRLDCP = 1.0_JPRB / (RALSDCP - RALVDCP)

      ! Note: Defined in module/yoecldp.F90
      ! NCLDQL=1    ! liquid cloud water
      ! NCLDQI=2    ! ice cloud water
      ! NCLDQR=3    ! rain water
      ! NCLDQS=4    ! snow
      ! NCLDQV=5    ! vapour

      ! -----------------------------------------------
      ! Define species phase, 0=vapour, 1=liquid, 2=ice
      ! -----------------------------------------------
      IPHASE(NCLDQV) = 0
      IPHASE(NCLDQL) = 1
      IPHASE(NCLDQR) = 1
      IPHASE(NCLDQI) = 2
      IPHASE(NCLDQS) = 2

      ! ---------------------------------------------------
      ! Set up melting/freezing index,
      ! if an ice category melts/freezes, where does it go?
      ! ---------------------------------------------------
      IMELT(NCLDQV) = -99
      IMELT(NCLDQL) = NCLDQI
      IMELT(NCLDQR) = NCLDQS
      IMELT(NCLDQI) = NCLDQR
      IMELT(NCLDQS) = NCLDQR

      ! -----------------------------------------------
      ! INITIALIZATION OF OUTPUT TENDENCIES
      ! -----------------------------------------------
      DO JK=1,KLEV
        TENDENCY_LOC_T(JL, JK) = 0.0_JPRB
        TENDENCY_LOC_Q(JL, JK) = 0.0_JPRB
        TENDENCY_LOC_A(JL, JK) = 0.0_JPRB
      END DO
      DO JM=1,NCLV - 1
        DO JK=1,KLEV
          TENDENCY_LOC_CLD(JL, JK, JM) = 0.0_JPRB
        END DO
      END DO

      !-- These were uninitialized : meaningful only when we compare error differences
      DO JK=1,KLEV
        PCOVPTOT(JL, JK) = 0.0_JPRB
        TENDENCY_LOC_CLD(JL, JK, NCLV) = 0.0_JPRB
      END DO

      ! -------------------------
      ! set up fall speeds in m/s
      ! -------------------------
      ZVQX(NCLDQV) = 0.0_JPRB
      ZVQX(NCLDQL) = 0.0_JPRB
      ZVQX(NCLDQI) = YRECLDP%RVICE
      ZVQX(NCLDQR) = YRECLDP%RVRAIN
      ZVQX(NCLDQS) = YRECLDP%RVSNOW
      LLFALL(:) = .false.
      DO JM=1,NCLV
        IF (ZVQX(JM) > 0.0_JPRB)         LLFALL(JM) = .true.
        ! falling species
      END DO
      ! Set LLFALL to false for ice (but ice still sediments!)
      ! Need to rationalise this at some point
      LLFALL(NCLDQI) = .false.


      !######################################################################
      !             1.  *** INITIAL VALUES FOR VARIABLES ***
      !######################################################################


      ! ----------------------
      ! non CLV initialization
      ! ----------------------
      DO JK=1,KLEV
        ZTP1(JL, JK) = PT(JL, JK) + PTSPHY*TENDENCY_TMP_T(JL, JK)
        ZQX(JL, JK, NCLDQV) = PQ(JL, JK) + PTSPHY*TENDENCY_TMP_Q(JL, JK)
        ZQX0(JL, JK, NCLDQV) = PQ(JL, JK) + PTSPHY*TENDENCY_TMP_Q(JL, JK)
        ZA(JL, JK) = PA(JL, JK) + PTSPHY*TENDENCY_TMP_A(JL, JK)
        ZAORIG(JL, JK) = PA(JL, JK) + PTSPHY*TENDENCY_TMP_A(JL, JK)
      END DO

      ! -------------------------------------
      ! initialization for CLV family
      ! -------------------------------------
      DO JM=1,NCLV - 1
        DO JK=1,KLEV
          ZQX(JL, JK, JM) = PCLV(JL, JK, JM) + PTSPHY*TENDENCY_TMP_CLD(JL, JK, JM)
          ZQX0(JL, JK, JM) = PCLV(JL, JK, JM) + PTSPHY*TENDENCY_TMP_CLD(JL, JK, JM)
        END DO
      END DO

      !-------------
      ! zero arrays
      !-------------
      DO JM=1,NCLV
        DO JK=1,KLEV + 1
          ZPFPLSX(JL, JK, JM) = 0.0_JPRB            ! precip fluxes
        END DO
      END DO

      DO JM=1,NCLV
        DO JK=1,KLEV
          ZQXN2D(JL, JK, JM) = 0.0_JPRB            ! end of timestep values in 2D
          ZLNEG(JL, JK, JM) = 0.0_JPRB            ! negative input check
        END DO
      END DO

      PRAINFRAC_TOPRFZ(JL) = 0.0_JPRB        ! rain fraction at top of refreezing layer
      LLRAINLIQ = .true.        ! Assume all raindrops are liquid initially

      ! ----------------------------------------------------
      ! Tidy up very small cloud cover or total cloud water
      ! ----------------------------------------------------
      DO JK=1,KLEV
        IF (ZQX(JL, JK, NCLDQL) + ZQX(JL, JK, NCLDQI) < YRECLDP%RLMIN .or. ZA(JL, JK) < YRECLDP%RAMIN) THEN

          ! Evaporate small cloud liquid water amounts
          ZLNEG(JL, JK, NCLDQL) = ZLNEG(JL, JK, NCLDQL) + ZQX(JL, JK, NCLDQL)
          ZQADJ = ZQX(JL, JK, NCLDQL)*ZQTMST
          TENDENCY_LOC_Q(JL, JK) = TENDENCY_LOC_Q(JL, JK) + ZQADJ
          TENDENCY_LOC_T(JL, JK) = TENDENCY_LOC_T(JL, JK) - RALVDCP*ZQADJ
          ZQX(JL, JK, NCLDQV) = ZQX(JL, JK, NCLDQV) + ZQX(JL, JK, NCLDQL)
          ZQX(JL, JK, NCLDQL) = 0.0_JPRB

          ! Evaporate small cloud ice water amounts
          ZLNEG(JL, JK, NCLDQI) = ZLNEG(JL, JK, NCLDQI) + ZQX(JL, JK, NCLDQI)
          ZQADJ = ZQX(JL, JK, NCLDQI)*ZQTMST
          TENDENCY_LOC_Q(JL, JK) = TENDENCY_LOC_Q(JL, JK) + ZQADJ
          TENDENCY_LOC_T(JL, JK) = TENDENCY_LOC_T(JL, JK) - RALSDCP*ZQADJ
          ZQX(JL, JK, NCLDQV) = ZQX(JL, JK, NCLDQV) + ZQX(JL, JK, NCLDQI)
          ZQX(JL, JK, NCLDQI) = 0.0_JPRB

          ! Set cloud cover to zero
          ZA(JL, JK) = 0.0_JPRB

        END IF
      END DO

      ! ---------------------------------
      ! Tidy up small CLV variables
      ! ---------------------------------
      !DIR$ IVDEP
      DO JM=1,NCLV - 1
        !DIR$ IVDEP
        DO JK=1,KLEV
          !DIR$ IVDEP
          IF (ZQX(JL, JK, JM) < YRECLDP%RLMIN) THEN
            ZLNEG(JL, JK, JM) = ZLNEG(JL, JK, JM) + ZQX(JL, JK, JM)
            ZQADJ = ZQX(JL, JK, JM)*ZQTMST
            TENDENCY_LOC_Q(JL, JK) = TENDENCY_LOC_Q(JL, JK) + ZQADJ
            IF (IPHASE(JM) == 1)             TENDENCY_LOC_T(JL, JK) = TENDENCY_LOC_T(JL, JK) - RALVDCP*ZQADJ
            IF (IPHASE(JM) == 2)             TENDENCY_LOC_T(JL, JK) = TENDENCY_LOC_T(JL, JK) - RALSDCP*ZQADJ
            ZQX(JL, JK, NCLDQV) = ZQX(JL, JK, NCLDQV) + ZQX(JL, JK, JM)
            ZQX(JL, JK, JM) = 0.0_JPRB
          END IF
        END DO
      END DO


      ! ------------------------------
      ! Define saturation values
      ! ------------------------------
      DO JK=1,KLEV
        !----------------------------------------
        ! old *diagnostic* mixed phase saturation
        !----------------------------------------
        ZFOEALFA(JL, JK) = FOEALFA(ZTP1(JL, JK))
        ZFOEEWMT(JL, JK) = MIN(FOEEWM(ZTP1(JL, JK)) / PAP(JL, JK), 0.5_JPRB)
        ZQSMIX(JL, JK) = ZFOEEWMT(JL, JK)
        ZQSMIX(JL, JK) = ZQSMIX(JL, JK) / (1.0_JPRB - RETV*ZQSMIX(JL, JK))

        !---------------------------------------------
        ! ice saturation T<273K
        ! liquid water saturation for T>273K
        !---------------------------------------------
        ZALFA = FOEDELTA(ZTP1(JL, JK))
        ZFOEEW(JL, JK) = MIN((ZALFA*FOEELIQ(ZTP1(JL, JK)) + (1.0_JPRB - ZALFA)*FOEEICE(ZTP1(JL, JK))) / PAP(JL, JK), 0.5_JPRB)
        ZFOEEW(JL, JK) = MIN(0.5_JPRB, ZFOEEW(JL, JK))
        ZQSICE(JL, JK) = ZFOEEW(JL, JK) / (1.0_JPRB - RETV*ZFOEEW(JL, JK))

        !----------------------------------
        ! liquid water saturation
        !----------------------------------
        ZFOEELIQT(JL, JK) = MIN(FOEELIQ(ZTP1(JL, JK)) / PAP(JL, JK), 0.5_JPRB)
        ZQSLIQ(JL, JK) = ZFOEELIQT(JL, JK)
        ZQSLIQ(JL, JK) = ZQSLIQ(JL, JK) / (1.0_JPRB - RETV*ZQSLIQ(JL, JK))

        !   !----------------------------------
        !   ! ice water saturation
        !   !----------------------------------
        !   ZFOEEICET(JL,JK)=MIN(FOEEICE(ZTP1(JL,JK))/PAP(JL,JK),0.5_JPRB)
        !   ZQSICE(JL,JK)=ZFOEEICET(JL,JK)
        !   ZQSICE(JL,JK)=ZQSICE(JL,JK)/(1.0_JPRB-RETV*ZQSICE(JL,JK))

      END DO

      DO JK=1,KLEV


        !------------------------------------------
        ! Ensure cloud fraction is between 0 and 1
        !------------------------------------------
        ZA(JL, JK) = MAX(0.0_JPRB, MIN(1.0_JPRB, ZA(JL, JK)))

        !-------------------------------------------------------------------
        ! Calculate liq/ice fractions (no longer a diagnostic relationship)
        !-------------------------------------------------------------------
        ZLI(JL, JK) = ZQX(JL, JK, NCLDQL) + ZQX(JL, JK, NCLDQI)
        IF (ZLI(JL, JK) > YRECLDP%RLMIN) THEN
          ZLIQFRAC(JL, JK) = ZQX(JL, JK, NCLDQL) / ZLI(JL, JK)
          ZICEFRAC(JL, JK) = 1.0_JPRB - ZLIQFRAC(JL, JK)
        ELSE
          ZLIQFRAC(JL, JK) = 0.0_JPRB
          ZICEFRAC(JL, JK) = 0.0_JPRB
        END IF

      END DO

      !######################################################################
      !        2.       *** CONSTANTS AND PARAMETERS ***
      !######################################################################
      !  Calculate L in updrafts of bl-clouds
      !  Specify QS, P/PS for tropopause (for c2)
      !  And initialize variables
      !------------------------------------------

      !---------------------------------
      ! Find tropopause level (ZTRPAUS)
      !---------------------------------
      ZTRPAUS = 0.1_JPRB
      ZPAPHD = 1.0_JPRB / PAPH(JL, KLEV + 1)
      DO JK=1,KLEV - 1
        ZSIG = PAP(JL, JK)*ZPAPHD
        IF (ZSIG > 0.1_JPRB .and. ZSIG < 0.4_JPRB .and. ZTP1(JL, JK) > ZTP1(JL, JK + 1)) THEN
          ZTRPAUS = ZSIG
        END IF
      END DO

      !-----------------------------
      ! Reset single level variables
      !-----------------------------

      ZANEWM1 = 0.0_JPRB
      ZDA = 0.0_JPRB
      ZCOVPCLR = 0.0_JPRB
      ZCOVPMAX = 0.0_JPRB
      ZCOVPTOT = 0.0_JPRB
      ZCLDTOPDIST = 0.0_JPRB

      !######################################################################
      !           3.       *** PHYSICS ***
      !######################################################################


      !----------------------------------------------------------------------
      !                       START OF VERTICAL LOOP
      !----------------------------------------------------------------------

      DO JK=YRECLDP%NCLDTOP,KLEV

        !----------------------------------------------------------------------
        ! 3.0 INITIALIZE VARIABLES
        !----------------------------------------------------------------------

        !---------------------------------
        ! First guess microphysics
        !---------------------------------
        DO JM=1,NCLV
          ZQXFG(JM) = ZQX(JL, JK, JM)
        END DO

        !---------------------------------
        ! Set KLON arrays to zero
        !---------------------------------

        ZLICLD = 0.0_JPRB
        ZRAINAUT = 0.0_JPRB          ! currently needed for diags
        ZRAINACC = 0.0_JPRB          ! currently needed for diags
        ZSNOWAUT = 0.0_JPRB          ! needed
        ZLDEFR = 0.0_JPRB
        ZACUST = 0.0_JPRB          ! set later when needed
        ZQPRETOT = 0.0_JPRB
        ZLFINALSUM = 0.0_JPRB

        ! Required for first guess call
        ZLCOND1 = 0.0_JPRB
        ZLCOND2 = 0.0_JPRB
        ZSUPSAT = 0.0_JPRB
        ZLEVAPL = 0.0_JPRB
        ZLEVAPI = 0.0_JPRB

        !-------------------------------------
        ! solvers for cloud fraction
        !-------------------------------------
        ZSOLAB = 0.0_JPRB
        ZSOLAC = 0.0_JPRB

        ZICETOT = 0.0_JPRB

        !------------------------------------------
        ! reset matrix so missing pathways are set
        !------------------------------------------
        DO JM=1,NCLV
          DO JN=1,NCLV
            ZSOLQB(JN, JM) = 0.0_JPRB
            ZSOLQA(JN, JM) = 0.0_JPRB
          END DO
        END DO

        !----------------------------------
        ! reset new microphysics variables
        !----------------------------------
        DO JM=1,NCLV
          ZFALLSRCE(JM) = 0.0_JPRB
          ZFALLSINK(JM) = 0.0_JPRB
          ZCONVSRCE(JM) = 0.0_JPRB
          ZCONVSINK(JM) = 0.0_JPRB
          ZPSUPSATSRCE(JM) = 0.0_JPRB
          ZRATIO(JM) = 0.0_JPRB
        END DO


        !-------------------------
        ! derived variables needed
        !-------------------------

        ZDP = PAPH(JL, JK + 1) - PAPH(JL, JK)          ! dp
        ZGDP = RG / ZDP          ! g/dp
        ZRHO = PAP(JL, JK) / ((RD*ZTP1(JL, JK)))          ! p/RT air density

        ZDTGDP = PTSPHY*ZGDP          ! dt g/dp
        ZRDTGDP = ZDP*(1.0_JPRB / ((PTSPHY*RG)))          ! 1/(dt g/dp)

        IF (JK > 1)         ZDTGDPF = (PTSPHY*RG) / (PAP(JL, JK) - PAP(JL, JK - 1))

        !------------------------------------
        ! Calculate dqs/dT correction factor
        !------------------------------------
        ! Reminder: RETV=RV/RD-1

        ! liquid
        ZFACW = R5LES / ((ZTP1(JL, JK) - R4LES)**2)
        ZCOR = 1.0_JPRB / (1.0_JPRB - RETV*ZFOEELIQT(JL, JK))
        ZDQSLIQDT = ZFACW*ZCOR*ZQSLIQ(JL, JK)
        ZCORQSLIQ = 1.0_JPRB + RALVDCP*ZDQSLIQDT

        ! ice
        ZFACI = R5IES / ((ZTP1(JL, JK) - R4IES)**2)
        ZCOR = 1.0_JPRB / (1.0_JPRB - RETV*ZFOEEW(JL, JK))
        ZDQSICEDT = ZFACI*ZCOR*ZQSICE(JL, JK)
        ZCORQSICE = 1.0_JPRB + RALSDCP*ZDQSICEDT

        ! diagnostic mixed
        ZALFAW = ZFOEALFA(JL, JK)
        ZALFAWM = ZALFAW
        ZFAC = ZALFAW*ZFACW + (1.0_JPRB - ZALFAW)*ZFACI
        ZCOR = 1.0_JPRB / (1.0_JPRB - RETV*ZFOEEWMT(JL, JK))
        ZDQSMIXDT = ZFAC*ZCOR*ZQSMIX(JL, JK)
        ZCORQSMIX = 1.0_JPRB + FOELDCPM(ZTP1(JL, JK))*ZDQSMIXDT

        ! evaporation/sublimation limits
        ZEVAPLIMMIX = MAX((ZQSMIX(JL, JK) - ZQX(JL, JK, NCLDQV)) / ZCORQSMIX, 0.0_JPRB)
        ZEVAPLIMLIQ = MAX((ZQSLIQ(JL, JK) - ZQX(JL, JK, NCLDQV)) / ZCORQSLIQ, 0.0_JPRB)
        ZEVAPLIMICE = MAX((ZQSICE(JL, JK) - ZQX(JL, JK, NCLDQV)) / ZCORQSICE, 0.0_JPRB)

        !--------------------------------
        ! in-cloud consensate amount
        !--------------------------------
        ZTMPA = 1.0_JPRB / MAX(ZA(JL, JK), ZEPSEC)
        ZLIQCLD = ZQX(JL, JK, NCLDQL)*ZTMPA
        ZICECLD = ZQX(JL, JK, NCLDQI)*ZTMPA
        ZLICLD = ZLIQCLD + ZICECLD


        !------------------------------------------------
        ! Evaporate very small amounts of liquid and ice
        !------------------------------------------------

        IF (ZQX(JL, JK, NCLDQL) < YRECLDP%RLMIN) THEN
          ZSOLQA(NCLDQV, NCLDQL) = ZQX(JL, JK, NCLDQL)
          ZSOLQA(NCLDQL, NCLDQV) = -ZQX(JL, JK, NCLDQL)
        END IF

        IF (ZQX(JL, JK, NCLDQI) < YRECLDP%RLMIN) THEN
          ZSOLQA(NCLDQV, NCLDQI) = ZQX(JL, JK, NCLDQI)
          ZSOLQA(NCLDQI, NCLDQV) = -ZQX(JL, JK, NCLDQI)
        END IF


        !---------------------------------------------------------------------
        !  3.1  ICE SUPERSATURATION ADJUSTMENT
        !---------------------------------------------------------------------
        ! Note that the supersaturation adjustment is made with respect to
        ! liquid saturation:  when T>0C
        ! ice saturation:     when T<0C
        !                     with an adjustment made to allow for ice
        !                     supersaturation in the clear sky
        ! Note also that the KOOP factor automatically clips the supersaturation
        ! to a maximum set by the liquid water saturation mixing ratio
        ! important for temperatures near to but below 0C
        !-----------------------------------------------------------------------

        !DIR$ NOFUSION

        !-----------------------------------
        ! 3.1.1 Supersaturation limit (from Koop)
        !-----------------------------------
        ! Needs to be set for all temperatures
        ZFOKOOP = FOKOOP(ZTP1(JL, JK))

        IF (ZTP1(JL, JK) >= RTT .or. YRECLDP%NSSOPT == 0) THEN
          ZFAC = 1.0_JPRB
          ZFACI = 1.0_JPRB
        ELSE
          ZFAC = ZA(JL, JK) + ZFOKOOP*(1.0_JPRB - ZA(JL, JK))
          ZFACI = PTSPHY / YRECLDP%RKOOPTAU
        END IF

        !-------------------------------------------------------------------
        ! 3.1.2 Calculate supersaturation wrt Koop including dqs/dT
        !       correction factor
        ! [#Note: QSICE or QSLIQ]
        !-------------------------------------------------------------------

        ! Calculate supersaturation to add to cloud
        IF (ZA(JL, JK) > 1.0_JPRB - YRECLDP%RAMIN) THEN
          ZSUPSAT = MAX((ZQX(JL, JK, NCLDQV) - ZFAC*ZQSICE(JL, JK)) / ZCORQSICE, 0.0_JPRB)
        ELSE
          ! Calculate environmental humidity supersaturation
          ZQP1ENV = (ZQX(JL, JK, NCLDQV) - ZA(JL, JK)*ZQSICE(JL, JK)) / MAX(1.0_JPRB - ZA(JL, JK), ZEPSILON)
          !& SIGN(MAX(ABS(1.0_JPRB-ZA(JL,JK)),ZEPSILON),1.0_JPRB-ZA(JL,JK))
          ZSUPSAT = MAX(((1.0_JPRB - ZA(JL, JK))*(ZQP1ENV - ZFAC*ZQSICE(JL, JK))) / ZCORQSICE, 0.0_JPRB)
        END IF

        !-------------------------------------------------------------------
        ! Here the supersaturation is turned into liquid water
        ! However, if the temperature is below the threshold for homogeneous
        ! freezing then the supersaturation is turned instantly to ice.
        !--------------------------------------------------------------------

        IF (ZSUPSAT > ZEPSEC) THEN

          IF (ZTP1(JL, JK) > YRECLDP%RTHOMO) THEN
            ! Turn supersaturation into liquid water
            ZSOLQA(NCLDQL, NCLDQV) = ZSOLQA(NCLDQL, NCLDQV) + ZSUPSAT
            ZSOLQA(NCLDQV, NCLDQL) = ZSOLQA(NCLDQV, NCLDQL) - ZSUPSAT
            ! Include liquid in first guess
            ZQXFG(NCLDQL) = ZQXFG(NCLDQL) + ZSUPSAT
          ELSE
            ! Turn supersaturation into ice water
            ZSOLQA(NCLDQI, NCLDQV) = ZSOLQA(NCLDQI, NCLDQV) + ZSUPSAT
            ZSOLQA(NCLDQV, NCLDQI) = ZSOLQA(NCLDQV, NCLDQI) - ZSUPSAT
            ! Add ice to first guess for deposition term
            ZQXFG(NCLDQI) = ZQXFG(NCLDQI) + ZSUPSAT
          END IF

          ! Increase cloud amount using RKOOPTAU timescale
          ZSOLAC = (1.0_JPRB - ZA(JL, JK))*ZFACI

        END IF

        !-------------------------------------------------------
        ! 3.1.3 Include supersaturation from previous timestep
        ! (Calculated in sltENDIF semi-lagrangian LDSLPHY=T)
        !-------------------------------------------------------
        IF (PSUPSAT(JL, JK) > ZEPSEC) THEN
          IF (ZTP1(JL, JK) > YRECLDP%RTHOMO) THEN
            ! Turn supersaturation into liquid water
            ZSOLQA(NCLDQL, NCLDQL) = ZSOLQA(NCLDQL, NCLDQL) + PSUPSAT(JL, JK)
            ZPSUPSATSRCE(NCLDQL) = PSUPSAT(JL, JK)
            ! Add liquid to first guess for deposition term
            ZQXFG(NCLDQL) = ZQXFG(NCLDQL) + PSUPSAT(JL, JK)
            ! Store cloud budget diagnostics if required
          ELSE
            ! Turn supersaturation into ice water
            ZSOLQA(NCLDQI, NCLDQI) = ZSOLQA(NCLDQI, NCLDQI) + PSUPSAT(JL, JK)
            ZPSUPSATSRCE(NCLDQI) = PSUPSAT(JL, JK)
            ! Add ice to first guess for deposition term
            ZQXFG(NCLDQI) = ZQXFG(NCLDQI) + PSUPSAT(JL, JK)
            ! Store cloud budget diagnostics if required
          END IF

          ! Increase cloud amount using RKOOPTAU timescale
          ZSOLAC = (1.0_JPRB - ZA(JL, JK))*ZFACI
          ! Store cloud budget diagnostics if required
        END IF

        ! on JL

        !---------------------------------------------------------------------
        !  3.2  DETRAINMENT FROM CONVECTION
        !---------------------------------------------------------------------
        ! * Diagnostic T-ice/liq split retained for convection
        !    Note: This link is now flexible and a future convection
        !    scheme can detrain explicit seperate budgets of:
        !    cloud water, ice, rain and snow
        ! * There is no (1-ZA) multiplier term on the cloud detrainment
        !    term, since is now written in mass-flux terms
        ! [#Note: Should use ZFOEALFACU used in convection rather than ZFOEALFA]
        !---------------------------------------------------------------------
        IF (JK < KLEV .and. JK >= YRECLDP%NCLDTOP) THEN


          PLUDE(JL, JK) = PLUDE(JL, JK)*ZDTGDP

          IF (LDCUM(JL) .and. PLUDE(JL, JK) > YRECLDP%RLMIN .and. PLU(JL, JK + 1) > ZEPSEC) THEN

            ZSOLAC = ZSOLAC + PLUDE(JL, JK) / PLU(JL, JK + 1)
            ! *diagnostic temperature split*
            ZALFAW = ZFOEALFA(JL, JK)
            ZCONVSRCE(NCLDQL) = ZALFAW*PLUDE(JL, JK)
            ZCONVSRCE(NCLDQI) = (1.0_JPRB - ZALFAW)*PLUDE(JL, JK)
            ZSOLQA(NCLDQL, NCLDQL) = ZSOLQA(NCLDQL, NCLDQL) + ZCONVSRCE(NCLDQL)
            ZSOLQA(NCLDQI, NCLDQI) = ZSOLQA(NCLDQI, NCLDQI) + ZCONVSRCE(NCLDQI)

          ELSE

            PLUDE(JL, JK) = 0.0_JPRB

          END IF
          ! *convective snow detrainment source
          IF (LDCUM(JL))           ZSOLQA(NCLDQS, NCLDQS) = ZSOLQA(NCLDQS, NCLDQS) + PSNDE(JL, JK)*ZDTGDP


        END IF
        ! JK<KLEV

        !---------------------------------------------------------------------
        !  3.3  SUBSIDENCE COMPENSATING CONVECTIVE UPDRAUGHTS
        !---------------------------------------------------------------------
        ! Three terms:
        ! * Convective subsidence source of cloud from layer above
        ! * Evaporation of cloud within the layer
        ! * Subsidence sink of cloud to the layer below (Implicit solution)
        !---------------------------------------------------------------------

        !-----------------------------------------------
        ! Subsidence source from layer above
        !               and
        ! Evaporation of cloud within the layer
        !-----------------------------------------------
        IF (JK > YRECLDP%NCLDTOP) THEN

          ZMF = MAX(0.0_JPRB, (PMFU(JL, JK) + PMFD(JL, JK))*ZDTGDP)
          ZACUST = ZMF*ZANEWM1

          DO JM=1,NCLV
            IF (.not.LLFALL(JM) .and. IPHASE(JM) > 0) THEN
              ZLCUST(JM) = ZMF*ZQXNM1(JM)
              ! record total flux for enthalpy budget:
              ZCONVSRCE(JM) = ZCONVSRCE(JM) + ZLCUST(JM)
            END IF
          END DO

          ! Now have to work out how much liquid evaporates at arrival point
          ! since there is no prognostic memory for in-cloud humidity, i.e.
          ! we always assume cloud is saturated.

          ZDTDP = (ZRDCP*0.5_JPRB*(ZTP1(JL, JK - 1) + ZTP1(JL, JK))) / PAPH(JL, JK)
          ZDTFORC = ZDTDP*(PAP(JL, JK) - PAP(JL, JK - 1))
          ![#Note: Diagnostic mixed phase should be replaced below]
          ZDQS = ZANEWM1*ZDTFORC*ZDQSMIXDT

          DO JM=1,NCLV
            IF (.not.LLFALL(JM) .and. IPHASE(JM) > 0) THEN
              ZLFINAL = MAX(0.0_JPRB, ZLCUST(JM) - ZDQS)                !lim to zero
              ! no supersaturation allowed incloud ---V
              ZEVAP = MIN((ZLCUST(JM) - ZLFINAL), ZEVAPLIMMIX)
              !          ZEVAP=0.0_JPRB
              ZLFINAL = ZLCUST(JM) - ZEVAP
              ZLFINALSUM = ZLFINALSUM + ZLFINAL                ! sum

              ZSOLQA(JM, JM) = ZSOLQA(JM, JM) + ZLCUST(JM)                ! whole sum
              ZSOLQA(NCLDQV, JM) = ZSOLQA(NCLDQV, JM) + ZEVAP
              ZSOLQA(JM, NCLDQV) = ZSOLQA(JM, NCLDQV) - ZEVAP
            END IF
          END DO

          !  Reset the cloud contribution if no cloud water survives to this level:
          IF (ZLFINALSUM < ZEPSEC)           ZACUST = 0.0_JPRB
          ZSOLAC = ZSOLAC + ZACUST

        END IF
        ! on  JK>NCLDTOP

        !---------------------------------------------------------------------
        ! Subsidence sink of cloud to the layer below
        ! (Implicit - re. CFL limit on convective mass flux)
        !---------------------------------------------------------------------


        IF (JK < KLEV) THEN

          ZMFDN = MAX(0.0_JPRB, (PMFU(JL, JK + 1) + PMFD(JL, JK + 1))*ZDTGDP)

          ZSOLAB = ZSOLAB + ZMFDN
          ZSOLQB(NCLDQL, NCLDQL) = ZSOLQB(NCLDQL, NCLDQL) + ZMFDN
          ZSOLQB(NCLDQI, NCLDQI) = ZSOLQB(NCLDQI, NCLDQI) + ZMFDN

          ! Record sink for cloud budget and enthalpy budget diagnostics
          ZCONVSINK(NCLDQL) = ZMFDN
          ZCONVSINK(NCLDQI) = ZMFDN

        END IF


        !----------------------------------------------------------------------
        ! 3.4  EROSION OF CLOUDS BY TURBULENT MIXING
        !----------------------------------------------------------------------
        ! NOTE: In default tiedtke scheme this process decreases the cloud
        !       area but leaves the specific cloud water content
        !       within clouds unchanged
        !----------------------------------------------------------------------

        ! ------------------------------
        ! Define turbulent erosion rate
        ! ------------------------------
        ZLDIFDT = YRECLDP%RCLDIFF*PTSPHY          !original version
        !Increase by factor of 5 for convective points
        IF (KTYPE(JL) > 0 .and. PLUDE(JL, JK) > ZEPSEC)         ZLDIFDT = YRECLDP%RCLDIFF_CONVI*ZLDIFDT

        ! At the moment, works on mixed RH profile and partitioned ice/liq fraction
        ! so that it is similar to previous scheme
        ! Should apply RHw for liquid cloud and RHi for ice cloud separately
        IF (ZLI(JL, JK) > ZEPSEC) THEN
          ! Calculate environmental humidity
          !      ZQE=(ZQX(JL,JK,NCLDQV)-ZA(JL,JK)*ZQSMIX(JL,JK))/&
          !    &      MAX(ZEPSEC,1.0_JPRB-ZA(JL,JK))
          !      ZE=ZLDIFDT(JL)*MAX(ZQSMIX(JL,JK)-ZQE,0.0_JPRB)
          ZE = ZLDIFDT*MAX(ZQSMIX(JL, JK) - ZQX(JL, JK, NCLDQV), 0.0_JPRB)
          ZLEROS = ZA(JL, JK)*ZE
          ZLEROS = MIN(ZLEROS, ZEVAPLIMMIX)
          ZLEROS = MIN(ZLEROS, ZLI(JL, JK))
          ZAEROS = ZLEROS / ZLICLD            !if linear term

          ! Erosion is -ve LINEAR in L,A
          ZSOLAC = ZSOLAC - ZAEROS            !linear

          ZSOLQA(NCLDQV, NCLDQL) = ZSOLQA(NCLDQV, NCLDQL) + ZLIQFRAC(JL, JK)*ZLEROS
          ZSOLQA(NCLDQL, NCLDQV) = ZSOLQA(NCLDQL, NCLDQV) - ZLIQFRAC(JL, JK)*ZLEROS
          ZSOLQA(NCLDQV, NCLDQI) = ZSOLQA(NCLDQV, NCLDQI) + ZICEFRAC(JL, JK)*ZLEROS
          ZSOLQA(NCLDQI, NCLDQV) = ZSOLQA(NCLDQI, NCLDQV) - ZICEFRAC(JL, JK)*ZLEROS

        END IF

        !----------------------------------------------------------------------
        ! 3.4  CONDENSATION/EVAPORATION DUE TO DQSAT/DT
        !----------------------------------------------------------------------
        !  calculate dqs/dt
        !  Note: For the separate prognostic Qi and Ql, one would ideally use
        !  Qsat/DT wrt liquid/Koop here, since the physics is that new clouds
        !  forms by liquid droplets [liq] or when aqueous aerosols [Koop] form.
        !  These would then instantaneous freeze if T<-38C or lead to ice growth
        !  by deposition in warmer mixed phase clouds.  However, since we do
        !  not have a separate prognostic equation for in-cloud humidity or a
        !  statistical scheme approach in place, the depositional growth of ice
        !  in the mixed phase can not be modelled and we resort to supersaturation
        !  wrt ice instanteously converting to ice over one timestep
        !  (see Tompkins et al. QJRMS 2007 for details)
        !  Thus for the initial implementation the diagnostic mixed phase is
        !  retained for the moment, and the level of approximation noted.
        !----------------------------------------------------------------------

        ZDTDP = (ZRDCP*ZTP1(JL, JK)) / PAP(JL, JK)
        ZDPMXDT = ZDP*ZQTMST
        ZMFDN = 0.0_JPRB
        IF (JK < KLEV)         ZMFDN = PMFU(JL, JK + 1) + PMFD(JL, JK + 1)
        ZWTOT = PVERVEL(JL, JK) + 0.5_JPRB*RG*(PMFU(JL, JK) + PMFD(JL, JK) + ZMFDN)
        ZWTOT = MIN(ZDPMXDT, MAX(-ZDPMXDT, ZWTOT))
        ZZZDT = PHRSW(JL, JK) + PHRLW(JL, JK)
        ZDTDIAB = MIN(ZDPMXDT*ZDTDP, MAX(-ZDPMXDT*ZDTDP, ZZZDT))*PTSPHY + RALFDCP*ZLDEFR
        ! Note: ZLDEFR should be set to the difference between the mixed phase functions
        ! in the convection and cloud scheme, but this is not calculated, so is zero and
        ! the functions must be the same
        ZDTFORC = ZDTDP*ZWTOT*PTSPHY + ZDTDIAB
        ZQOLD = ZQSMIX(JL, JK)
        ZTOLD = ZTP1(JL, JK)
        ZTP1(JL, JK) = ZTP1(JL, JK) + ZDTFORC
        ZTP1(JL, JK) = MAX(ZTP1(JL, JK), 160.0_JPRB)
        LLFLAG = .true.

        ! Formerly a call to CUADJTQ(..., ICALL=5)
        ZQP = 1.0_JPRB / PAP(JL, JK)
        ZQSAT = FOEEWM(ZTP1(JL, JK))*ZQP
        ZQSAT = MIN(0.5_JPRB, ZQSAT)
        ZCOR = 1.0_JPRB / (1.0_JPRB - RETV*ZQSAT)
        ZQSAT = ZQSAT*ZCOR
        ZCOND = (ZQSMIX(JL, JK) - ZQSAT) / (1.0_JPRB + ZQSAT*ZCOR*FOEDEM(ZTP1(JL, JK)))
        ZTP1(JL, JK) = ZTP1(JL, JK) + FOELDCPM(ZTP1(JL, JK))*ZCOND
        ZQSMIX(JL, JK) = ZQSMIX(JL, JK) - ZCOND
        ZQSAT = FOEEWM(ZTP1(JL, JK))*ZQP
        ZQSAT = MIN(0.5_JPRB, ZQSAT)
        ZCOR = 1.0_JPRB / (1.0_JPRB - RETV*ZQSAT)
        ZQSAT = ZQSAT*ZCOR
        ZCOND1 = (ZQSMIX(JL, JK) - ZQSAT) / (1.0_JPRB + ZQSAT*ZCOR*FOEDEM(ZTP1(JL, JK)))
        ZTP1(JL, JK) = ZTP1(JL, JK) + FOELDCPM(ZTP1(JL, JK))*ZCOND1
        ZQSMIX(JL, JK) = ZQSMIX(JL, JK) - ZCOND1

        ZDQS = ZQSMIX(JL, JK) - ZQOLD
        ZQSMIX(JL, JK) = ZQOLD
        ZTP1(JL, JK) = ZTOLD

        !----------------------------------------------------------------------
        ! 3.4a  ZDQS(JL) > 0:  EVAPORATION OF CLOUDS
        ! ----------------------------------------------------------------------
        ! Erosion term is LINEAR in L
        ! Changed to be uniform distribution in cloud region


        ! Previous function based on DELTA DISTRIBUTION in cloud:
        IF (ZDQS > 0.0_JPRB) THEN
          !    If subsidence evaporation term is turned off, then need to use updated
          !    liquid and cloud here?
          !    ZLEVAP = MAX(ZA(JL,JK)+ZACUST(JL),1.0_JPRB)*MIN(ZDQS(JL),ZLICLD(JL)+ZLFINALSUM(JL))
          ZLEVAP = ZA(JL, JK)*MIN(ZDQS, ZLICLD)
          ZLEVAP = MIN(ZLEVAP, ZEVAPLIMMIX)
          ZLEVAP = MIN(ZLEVAP, MAX(ZQSMIX(JL, JK) - ZQX(JL, JK, NCLDQV), 0.0_JPRB))

          ! For first guess call
          ZLEVAPL = ZLIQFRAC(JL, JK)*ZLEVAP
          ZLEVAPI = ZICEFRAC(JL, JK)*ZLEVAP

          ZSOLQA(NCLDQV, NCLDQL) = ZSOLQA(NCLDQV, NCLDQL) + ZLIQFRAC(JL, JK)*ZLEVAP
          ZSOLQA(NCLDQL, NCLDQV) = ZSOLQA(NCLDQL, NCLDQV) - ZLIQFRAC(JL, JK)*ZLEVAP

          ZSOLQA(NCLDQV, NCLDQI) = ZSOLQA(NCLDQV, NCLDQI) + ZICEFRAC(JL, JK)*ZLEVAP
          ZSOLQA(NCLDQI, NCLDQV) = ZSOLQA(NCLDQI, NCLDQV) - ZICEFRAC(JL, JK)*ZLEVAP

        END IF


        !----------------------------------------------------------------------
        ! 3.4b ZDQS(JL) < 0: FORMATION OF CLOUDS
        !----------------------------------------------------------------------
        ! (1) Increase of cloud water in existing clouds
        IF (ZA(JL, JK) > ZEPSEC .and. ZDQS <= -YRECLDP%RLMIN) THEN

          ZLCOND1 = MAX(-ZDQS, 0.0_JPRB)            !new limiter

          !old limiter (significantly improves upper tropospheric humidity rms)
          IF (ZA(JL, JK) > 0.99_JPRB) THEN
            ZCOR = 1.0_JPRB / (1.0_JPRB - RETV*ZQSMIX(JL, JK))
            ZCDMAX = (ZQX(JL, JK, NCLDQV) - ZQSMIX(JL, JK)) / (1.0_JPRB + ZCOR*ZQSMIX(JL, JK)*FOEDEM(ZTP1(JL, JK)))
          ELSE
            ZCDMAX = (ZQX(JL, JK, NCLDQV) - ZA(JL, JK)*ZQSMIX(JL, JK)) / ZA(JL, JK)
          END IF
          ZLCOND1 = MAX(MIN(ZLCOND1, ZCDMAX), 0.0_JPRB)
          ! end old limiter

          ZLCOND1 = ZA(JL, JK)*ZLCOND1
          IF (ZLCOND1 < YRECLDP%RLMIN)           ZLCOND1 = 0.0_JPRB

          !-------------------------------------------------------------------------
          ! All increase goes into liquid unless so cold cloud homogeneously freezes
          ! Include new liquid formation in first guess value, otherwise liquid
          ! remains at cold temperatures until next timestep.
          !-------------------------------------------------------------------------
          IF (ZTP1(JL, JK) > YRECLDP%RTHOMO) THEN
            ZSOLQA(NCLDQL, NCLDQV) = ZSOLQA(NCLDQL, NCLDQV) + ZLCOND1
            ZSOLQA(NCLDQV, NCLDQL) = ZSOLQA(NCLDQV, NCLDQL) - ZLCOND1
            ZQXFG(NCLDQL) = ZQXFG(NCLDQL) + ZLCOND1
          ELSE
            ZSOLQA(NCLDQI, NCLDQV) = ZSOLQA(NCLDQI, NCLDQV) + ZLCOND1
            ZSOLQA(NCLDQV, NCLDQI) = ZSOLQA(NCLDQV, NCLDQI) - ZLCOND1
            ZQXFG(NCLDQI) = ZQXFG(NCLDQI) + ZLCOND1
          END IF
        END IF

        ! (2) Generation of new clouds (da/dt>0)


        IF (ZDQS <= -YRECLDP%RLMIN .and. ZA(JL, JK) < 1.0_JPRB - ZEPSEC) THEN

          !---------------------------
          ! Critical relative humidity
          !---------------------------
          ZRHC = YRECLDP%RAMID
          ZSIGK = PAP(JL, JK) / PAPH(JL, KLEV + 1)
          ! Increase RHcrit to 1.0 towards the surface (eta>0.8)
          IF (ZSIGK > 0.8_JPRB) THEN
            ZRHC = YRECLDP%RAMID + (1.0_JPRB - YRECLDP%RAMID)*((ZSIGK - 0.8_JPRB) / 0.2_JPRB)**2
          END IF

          ! Commented out for CY37R1 to reduce humidity in high trop and strat
          !      ! Increase RHcrit to 1.0 towards the tropopause (trop-0.2) and above
          !      ZBOTT=ZTRPAUS(JL)+0.2_JPRB
          !      IF(ZSIGK < ZBOTT) THEN
          !        ZRHC=RAMID+(1.0_JPRB-RAMID)*MIN(((ZBOTT-ZSIGK)/0.2_JPRB)**2,1.0_JPRB)
          !      ENDIF

          !---------------------------
          ! Supersaturation options
          !---------------------------
          IF (YRECLDP%NSSOPT == 0) THEN
            ! No scheme
            ZQE = (ZQX(JL, JK, NCLDQV) - ZA(JL, JK)*ZQSICE(JL, JK)) / MAX(ZEPSEC, 1.0_JPRB - ZA(JL, JK))
            ZQE = MAX(0.0_JPRB, ZQE)
          ELSE IF (YRECLDP%NSSOPT == 1) THEN
            ! Tompkins
            ZQE = (ZQX(JL, JK, NCLDQV) - ZA(JL, JK)*ZQSICE(JL, JK)) / MAX(ZEPSEC, 1.0_JPRB - ZA(JL, JK))
            ZQE = MAX(0.0_JPRB, ZQE)
          ELSE IF (YRECLDP%NSSOPT == 2) THEN
            ! Lohmann and Karcher
            ZQE = ZQX(JL, JK, NCLDQV)
          ELSE IF (YRECLDP%NSSOPT == 3) THEN
            ! Gierens
            ZQE = ZQX(JL, JK, NCLDQV) + ZLI(JL, JK)
          END IF

          IF (ZTP1(JL, JK) >= RTT .or. YRECLDP%NSSOPT == 0) THEN
            ! No ice supersaturation allowed
            ZFAC = 1.0_JPRB
          ELSE
            ! Ice supersaturation
            ZFAC = ZFOKOOP
          END IF

          IF (ZQE >= ZRHC*ZQSICE(JL, JK)*ZFAC .and. ZQE < ZQSICE(JL, JK)*ZFAC) THEN
            ! note: not **2 on 1-a term if ZQE is used.
            ! Added correction term ZFAC to numerator 15/03/2010
            ZACOND = -((1.0_JPRB - ZA(JL, JK))*ZFAC*ZDQS) / MAX(2.0_JPRB*(ZFAC*ZQSICE(JL, JK) - ZQE), ZEPSEC)

            ZACOND = MIN(ZACOND, 1.0_JPRB - ZA(JL, JK))              !PUT THE LIMITER BACK

            ! Linear term:
            ! Added correction term ZFAC 15/03/2010
            ZLCOND2 = -ZFAC*ZDQS*0.5_JPRB*ZACOND              !mine linear

            ! new limiter formulation
            ZZDL = (2.0_JPRB*(ZFAC*ZQSICE(JL, JK) - ZQE)) / MAX(ZEPSEC, 1.0_JPRB - ZA(JL, JK))
            ! Added correction term ZFAC 15/03/2010
            IF (ZFAC*ZDQS < -ZZDL) THEN
              ! ZLCONDLIM=(ZA(JL,JK)-1.0_JPRB)*ZDQS(JL)-ZQSICE(JL,JK)+ZQX(JL,JK,NCLDQV)
              ZLCONDLIM = (ZA(JL, JK) - 1.0_JPRB)*ZFAC*ZDQS - ZFAC*ZQSICE(JL, JK) + ZQX(JL, JK, NCLDQV)
              ZLCOND2 = MIN(ZLCOND2, ZLCONDLIM)
            END IF
            ZLCOND2 = MAX(ZLCOND2, 0.0_JPRB)

            IF (ZLCOND2 < YRECLDP%RLMIN .or. (1.0_JPRB - ZA(JL, JK)) < ZEPSEC) THEN
              ZLCOND2 = 0.0_JPRB
              ZACOND = 0.0_JPRB
            END IF
            IF (ZLCOND2 == 0.0_JPRB)             ZACOND = 0.0_JPRB

            ! Large-scale generation is LINEAR in A and LINEAR in L
            ZSOLAC = ZSOLAC + ZACOND              !linear

            !------------------------------------------------------------------------
            ! All increase goes into liquid unless so cold cloud homogeneously freezes
            ! Include new liquid formation in first guess value, otherwise liquid
            ! remains at cold temperatures until next timestep.
            !------------------------------------------------------------------------
            IF (ZTP1(JL, JK) > YRECLDP%RTHOMO) THEN
              ZSOLQA(NCLDQL, NCLDQV) = ZSOLQA(NCLDQL, NCLDQV) + ZLCOND2
              ZSOLQA(NCLDQV, NCLDQL) = ZSOLQA(NCLDQV, NCLDQL) - ZLCOND2
              ZQXFG(NCLDQL) = ZQXFG(NCLDQL) + ZLCOND2
            ELSE
              ! homogeneous freezing
              ZSOLQA(NCLDQI, NCLDQV) = ZSOLQA(NCLDQI, NCLDQV) + ZLCOND2
              ZSOLQA(NCLDQV, NCLDQI) = ZSOLQA(NCLDQV, NCLDQI) - ZLCOND2
              ZQXFG(NCLDQI) = ZQXFG(NCLDQI) + ZLCOND2
            END IF

          END IF
        END IF

        !----------------------------------------------------------------------
        ! 3.7 Growth of ice by vapour deposition
        !----------------------------------------------------------------------
        ! Following Rotstayn et al. 2001:
        ! does not use the ice nuclei number from cloudaer.F90
        ! but rather a simple Meyers et al. 1992 form based on the
        ! supersaturation and assuming clouds are saturated with
        ! respect to liquid water (well mixed), (or Koop adjustment)
        ! Growth considered as sink of liquid water if present so
        ! Bergeron-Findeisen adjustment in autoconversion term no longer needed
        !----------------------------------------------------------------------

        !--------------------------------------------------------
        !-
        !- Ice deposition following Rotstayn et al. (2001)
        !-  (monodisperse ice particle size distribution)
        !-
        !--------------------------------------------------------
        IF (IDEPICE == 1) THEN


          !--------------------------------------------------------------
          ! Calculate distance from cloud top
          ! defined by cloudy layer below a layer with cloud frac <0.01
          ! ZDZ = ZDP(JL)/(ZRHO(JL)*RG)
          !--------------------------------------------------------------

          IF (ZA(JL, JK - 1) < YRECLDP%RCLDTOPCF .and. ZA(JL, JK) >= YRECLDP%RCLDTOPCF) THEN
            ZCLDTOPDIST = 0.0_JPRB
          ELSE
            ZCLDTOPDIST = ZCLDTOPDIST + ZDP / ((ZRHO*RG))
          END IF

          !--------------------------------------------------------------
          ! only treat depositional growth if liquid present. due to fact
          ! that can not model ice growth from vapour without additional
          ! in-cloud water vapour variable
          !--------------------------------------------------------------
          IF (ZTP1(JL, JK) < RTT .and. ZQXFG(NCLDQL) > YRECLDP%RLMIN) THEN
            ! T<273K

            ZVPICE = (FOEEICE(ZTP1(JL, JK))*RV) / RD
            ZVPLIQ = ZVPICE*ZFOKOOP
            ZICENUCLEI = 1000.0_JPRB*EXP((12.96_JPRB*(ZVPLIQ - ZVPICE)) / ZVPLIQ - 0.639_JPRB)

            !------------------------------------------------
            !   2.4e-2 is conductivity of air
            !   8.8 = 700**1/3 = density of ice to the third
            !------------------------------------------------
            ZADD = (RLSTT*(RLSTT / ((RV*ZTP1(JL, JK))) - 1.0_JPRB)) / ((2.4E-2_JPRB*ZTP1(JL, JK)))
            ZBDD = (RV*ZTP1(JL, JK)*PAP(JL, JK)) / ((2.21_JPRB*ZVPICE))
            ZCVDS = (7.8_JPRB*(ZICENUCLEI / ZRHO)**0.666_JPRB*(ZVPLIQ - ZVPICE)) / ((8.87_JPRB*(ZADD + ZBDD)*ZVPICE))

            !-----------------------------------------------------
            ! RICEINIT=1.E-12_JPRB is initial mass of ice particle
            !-----------------------------------------------------
            ZICE0 = MAX(ZICECLD, (ZICENUCLEI*YRECLDP%RICEINIT) / ZRHO)

            !------------------
            ! new value of ice:
            !------------------
            ZINEW = (0.666_JPRB*ZCVDS*PTSPHY + ZICE0**0.666_JPRB)**1.5_JPRB

            !---------------------------
            ! grid-mean deposition rate:
            !---------------------------
            ZDEPOS = MAX(ZA(JL, JK)*(ZINEW - ZICE0), 0.0_JPRB)

            !--------------------------------------------------------------------
            ! Limit deposition to liquid water amount
            ! If liquid is all frozen, ice would use up reservoir of water
            ! vapour in excess of ice saturation mixing ratio - However this
            ! can not be represented without a in-cloud humidity variable. Using
            ! the grid-mean humidity would imply a large artificial horizontal
            ! flux from the clear sky to the cloudy area. We thus rely on the
            ! supersaturation check to clean up any remaining supersaturation
            !--------------------------------------------------------------------
            ZDEPOS = MIN(ZDEPOS, ZQXFG(NCLDQL))              ! limit to liquid water amount

            !--------------------------------------------------------------------
            ! At top of cloud, reduce deposition rate near cloud top to account for
            ! small scale turbulent processes, limited ice nucleation and ice fallout
            !--------------------------------------------------------------------
            !      ZDEPOS = ZDEPOS*MIN(RDEPLIQREFRATE+ZCLDTOPDIST(JL)/RDEPLIQREFDEPTH,1.0_JPRB)
            ! Change to include dependence on ice nuclei concentration
            ! to increase deposition rate with decreasing temperatures
            ZINFACTOR = MIN(ZICENUCLEI / 15000._JPRB, 1.0_JPRB)
            ZDEPOS = ZDEPOS*MIN(ZINFACTOR + (1.0_JPRB - ZINFACTOR)*(YRECLDP%RDEPLIQREFRATE + ZCLDTOPDIST /  &
            & YRECLDP%RDEPLIQREFDEPTH), 1.0_JPRB)

            !--------------
            ! add to matrix
            !--------------
            ZSOLQA(NCLDQI, NCLDQL) = ZSOLQA(NCLDQI, NCLDQL) + ZDEPOS
            ZSOLQA(NCLDQL, NCLDQI) = ZSOLQA(NCLDQL, NCLDQI) - ZDEPOS
            ZQXFG(NCLDQI) = ZQXFG(NCLDQI) + ZDEPOS
            ZQXFG(NCLDQL) = ZQXFG(NCLDQL) - ZDEPOS

          END IF

          !--------------------------------------------------------
          !-
          !- Ice deposition assuming ice PSD
          !-
          !--------------------------------------------------------
        ELSE IF (IDEPICE == 2) THEN


          !--------------------------------------------------------------
          ! Calculate distance from cloud top
          ! defined by cloudy layer below a layer with cloud frac <0.01
          ! ZDZ = ZDP(JL)/(ZRHO(JL)*RG)
          !--------------------------------------------------------------

          IF (ZA(JL, JK - 1) < YRECLDP%RCLDTOPCF .and. ZA(JL, JK) >= YRECLDP%RCLDTOPCF) THEN
            ZCLDTOPDIST = 0.0_JPRB
          ELSE
            ZCLDTOPDIST = ZCLDTOPDIST + ZDP / ((ZRHO*RG))
          END IF

          !--------------------------------------------------------------
          ! only treat depositional growth if liquid present. due to fact
          ! that can not model ice growth from vapour without additional
          ! in-cloud water vapour variable
          !--------------------------------------------------------------
          IF (ZTP1(JL, JK) < RTT .and. ZQXFG(NCLDQL) > YRECLDP%RLMIN) THEN
            ! T<273K

            ZVPICE = (FOEEICE(ZTP1(JL, JK))*RV) / RD
            ZVPLIQ = ZVPICE*ZFOKOOP
            ZICENUCLEI = 1000.0_JPRB*EXP((12.96_JPRB*(ZVPLIQ - ZVPICE)) / ZVPLIQ - 0.639_JPRB)

            !-----------------------------------------------------
            ! RICEINIT=1.E-12_JPRB is initial mass of ice particle
            !-----------------------------------------------------
            ZICE0 = MAX(ZICECLD, (ZICENUCLEI*YRECLDP%RICEINIT) / ZRHO)

            ! Particle size distribution
            ZTCG = 1.0_JPRB
            ZFACX1I = 1.0_JPRB

            ZAPLUSB =  &
            & YRECLDP%RCL_APB1*ZVPICE - YRECLDP%RCL_APB2*ZVPICE*ZTP1(JL, JK) + PAP(JL, JK)*YRECLDP%RCL_APB3*ZTP1(JL, JK)**3._JPRB
            ZCORRFAC = (1.0_JPRB / ZRHO)**0.5_JPRB
            ZCORRFAC2 = ((ZTP1(JL, JK) / 273.0_JPRB)**1.5_JPRB)*(393.0_JPRB / (ZTP1(JL, JK) + 120.0_JPRB))

            ZPR02 = (ZRHO*ZICE0*YRECLDP%RCL_CONST1I) / ((ZTCG*ZFACX1I))

            ZTERM1 = ((ZVPLIQ - ZVPICE)*ZTP1(JL, JK)**2.0_JPRB*ZVPICE*ZCORRFAC2*ZTCG*YRECLDP%RCL_CONST2I*ZFACX1I) /  &
            & ((ZRHO*ZAPLUSB*ZVPICE))
            ZTERM2 = 0.65_JPRB*YRECLDP%RCL_CONST6I*ZPR02**YRECLDP%RCL_CONST4I +  &
            & (YRECLDP%RCL_CONST3I*ZCORRFAC**0.5_JPRB*ZRHO**0.5_JPRB*ZPR02**YRECLDP%RCL_CONST5I) / ZCORRFAC2**0.5_JPRB

            ZDEPOS = MAX(ZA(JL, JK)*ZTERM1*ZTERM2*PTSPHY, 0.0_JPRB)

            !--------------------------------------------------------------------
            ! Limit deposition to liquid water amount
            ! If liquid is all frozen, ice would use up reservoir of water
            ! vapour in excess of ice saturation mixing ratio - However this
            ! can not be represented without a in-cloud humidity variable. Using
            ! the grid-mean humidity would imply a large artificial horizontal
            ! flux from the clear sky to the cloudy area. We thus rely on the
            ! supersaturation check to clean up any remaining supersaturation
            !--------------------------------------------------------------------
            ZDEPOS = MIN(ZDEPOS, ZQXFG(NCLDQL))              ! limit to liquid water amount

            !--------------------------------------------------------------------
            ! At top of cloud, reduce deposition rate near cloud top to account for
            ! small scale turbulent processes, limited ice nucleation and ice fallout
            !--------------------------------------------------------------------
            ! Change to include dependence on ice nuclei concentration
            ! to increase deposition rate with decreasing temperatures
            ZINFACTOR = MIN(ZICENUCLEI / 15000._JPRB, 1.0_JPRB)
            ZDEPOS = ZDEPOS*MIN(ZINFACTOR + (1.0_JPRB - ZINFACTOR)*(YRECLDP%RDEPLIQREFRATE + ZCLDTOPDIST /  &
            & YRECLDP%RDEPLIQREFDEPTH), 1.0_JPRB)

            !--------------
            ! add to matrix
            !--------------
            ZSOLQA(NCLDQI, NCLDQL) = ZSOLQA(NCLDQI, NCLDQL) + ZDEPOS
            ZSOLQA(NCLDQL, NCLDQI) = ZSOLQA(NCLDQL, NCLDQI) - ZDEPOS
            ZQXFG(NCLDQI) = ZQXFG(NCLDQI) + ZDEPOS
            ZQXFG(NCLDQL) = ZQXFG(NCLDQL) - ZDEPOS
          END IF

        END IF
        ! on IDEPICE

        !######################################################################
        !              4  *** PRECIPITATION PROCESSES ***
        !######################################################################

        !----------------------------------
        ! revise in-cloud consensate amount
        !----------------------------------
        ZTMPA = 1.0_JPRB / MAX(ZA(JL, JK), ZEPSEC)
        ZLIQCLD = ZQXFG(NCLDQL)*ZTMPA
        ZICECLD = ZQXFG(NCLDQI)*ZTMPA
        ZLICLD = ZLIQCLD + ZICECLD

        !----------------------------------------------------------------------
        ! 4.2 SEDIMENTATION/FALLING OF *ALL* MICROPHYSICAL SPECIES
        !     now that rain, snow, graupel species are prognostic
        !     the precipitation flux can be defined directly level by level
        !     There is no vertical memory required from the flux variable
        !----------------------------------------------------------------------

        DO JM=1,NCLV
          IF (LLFALL(JM) .or. JM == NCLDQI) THEN
            !------------------------
            ! source from layer above
            !------------------------
            IF (JK > YRECLDP%NCLDTOP) THEN
              ZFALLSRCE(JM) = ZPFPLSX(JL, JK, JM)*ZDTGDP
              ZSOLQA(JM, JM) = ZSOLQA(JM, JM) + ZFALLSRCE(JM)
              ZQXFG(JM) = ZQXFG(JM) + ZFALLSRCE(JM)
              ! use first guess precip----------V
              ZQPRETOT = ZQPRETOT + ZQXFG(JM)
            END IF
            !-------------------------------------------------
            ! sink to next layer, constant fall speed
            !-------------------------------------------------
            ! if aerosol effect then override
            !  note that for T>233K this is the same as above.
            IF (YRECLDP%LAERICESED .and. JM == NCLDQI) THEN
              ZRE_ICE = PRE_ICE(JL, JK)
              ! The exponent value is from
              ! Morrison et al. JAS 2005 Appendix
              ZVQX(NCLDQI) = 0.002_JPRB*ZRE_ICE**1.0_JPRB
            END IF
            ZFALL = ZVQX(JM)*ZRHO
            !-------------------------------------------------
            ! modified by Heymsfield and Iaquinta JAS 2000
            !-------------------------------------------------
            ! ZFALL = ZFALL*((PAP(JL,JK)*RICEHI1)**(-0.178_JPRB)) &
            !            &*((ZTP1(JL,JK)*RICEHI2)**(-0.394_JPRB))

            ZFALLSINK(JM) = ZDTGDP*ZFALL
            ! Cloud budget diagnostic stored at end as implicit
            ! jl
          END IF
          ! LLFALL
        END DO
        ! jm

        !---------------------------------------------------------------
        ! Precip cover overlap using MAX-RAN Overlap
        ! Since precipitation is now prognostic we must
        !   1) apply an arbitrary minimum coverage (0.3) if precip>0
        !   2) abandon the 2-flux clr/cld treatment
        !   3) Thus, since we have no memory of the clear sky precip
        !      fraction, we mimic the previous method by reducing
        !      ZCOVPTOT(JL), which has the memory, proportionally with
        !      the precip evaporation rate, taking cloud fraction
        !      into account
        !   #3 above leads to much smoother vertical profiles of
        !   precipitation fraction than the Klein-Jakob scheme which
        !   monotonically increases precip fraction and then resets
        !   it to zero in a step function once clear-sky precip reaches
        !   zero.
        !---------------------------------------------------------------
        IF (ZQPRETOT > ZEPSEC) THEN
          ZCOVPTOT = 1.0_JPRB - ((1.0_JPRB - ZCOVPTOT)*(1.0_JPRB - MAX(ZA(JL, JK), ZA(JL, JK - 1)))) / (1.0_JPRB - MIN(ZA(JL, JK  &
          & - 1), 1.0_JPRB - 1.E-06_JPRB))
          ZCOVPTOT = MAX(ZCOVPTOT, YRECLDP%RCOVPMIN)
          ZCOVPCLR = MAX(0.0_JPRB, ZCOVPTOT - ZA(JL, JK))            ! clear sky proportion
          ZRAINCLD = ZQXFG(NCLDQR) / ZCOVPTOT
          ZSNOWCLD = ZQXFG(NCLDQS) / ZCOVPTOT
          ZCOVPMAX = MAX(ZCOVPTOT, ZCOVPMAX)
        ELSE
          ZRAINCLD = 0.0_JPRB
          ZSNOWCLD = 0.0_JPRB
          ZCOVPTOT = 0.0_JPRB            ! no flux - reset cover
          ZCOVPCLR = 0.0_JPRB            ! reset clear sky proportion
          ZCOVPMAX = 0.0_JPRB            ! reset max cover for ZZRH calc
        END IF

        !----------------------------------------------------------------------
        ! 4.3a AUTOCONVERSION TO SNOW
        !----------------------------------------------------------------------

        IF (ZTP1(JL, JK) <= RTT) THEN
          !-----------------------------------------------------
          !     Snow Autoconversion rate follow Lin et al. 1983
          !-----------------------------------------------------
          IF (ZICECLD > ZEPSEC) THEN

            ZZCO = PTSPHY*YRECLDP%RSNOWLIN1*EXP(YRECLDP%RSNOWLIN2*(ZTP1(JL, JK) - RTT))

            IF (YRECLDP%LAERICEAUTO) THEN
              ZLCRIT = PICRIT_AER(JL, JK)
              ! 0.3 = N**0.333 with N=0.027
              ZZCO = ZZCO*(YRECLDP%RNICE / PNICE(JL, JK))**0.333_JPRB
            ELSE
              ZLCRIT = YRECLDP%RLCRITSNOW
            END IF

            ZSNOWAUT = ZZCO*(1.0_JPRB - EXP(-(ZICECLD / ZLCRIT)**2))
            ZSOLQB(NCLDQS, NCLDQI) = ZSOLQB(NCLDQS, NCLDQI) + ZSNOWAUT

          END IF
        END IF

        !----------------------------------------------------------------------
        ! 4.3b AUTOCONVERSION WARM CLOUDS
        !   Collection and accretion will require separate treatment
        !   but for now we keep this simple treatment
        !----------------------------------------------------------------------

        IF (ZLIQCLD > ZEPSEC) THEN

          !--------------------------------------------------------
          !-
          !- Warm-rain process follow Sundqvist (1989)
          !-
          !--------------------------------------------------------
          IF (IWARMRAIN == 1) THEN

            ZZCO = YRECLDP%RKCONV*PTSPHY

            IF (YRECLDP%LAERLIQAUTOLSP) THEN
              ZLCRIT = PLCRIT_AER(JL, JK)
              ! 0.3 = N**0.333 with N=125 cm-3
              ZZCO = ZZCO*(YRECLDP%RCCN / PCCN(JL, JK))**0.333_JPRB
            ELSE
              ! Modify autoconversion threshold dependent on:
              !  land (polluted, high CCN, smaller droplets, higher threshold)
              !  sea  (clean, low CCN, larger droplets, lower threshold)
              IF (PLSM(JL) > 0.5_JPRB) THEN
                ZLCRIT = YRECLDP%RCLCRIT_LAND                  ! land
              ELSE
                ZLCRIT = YRECLDP%RCLCRIT_SEA                  ! ocean
              END IF
            END IF

            !------------------------------------------------------------------
            ! Parameters for cloud collection by rain and snow.
            ! Note that with new prognostic variable it is now possible
            ! to REPLACE this with an explicit collection parametrization
            !------------------------------------------------------------------
            ZPRECIP = (ZPFPLSX(JL, JK, NCLDQS) + ZPFPLSX(JL, JK, NCLDQR)) / MAX(ZEPSEC, ZCOVPTOT)
            ZCFPR = 1.0_JPRB + YRECLDP%RPRC1*SQRT(MAX(ZPRECIP, 0.0_JPRB))
            !      ZCFPR=1.0_JPRB + RPRC1*SQRT(MAX(ZPRECIP,0.0_JPRB))*&
            !       &ZCOVPTOT(JL)/(MAX(ZA(JL,JK),ZEPSEC))

            IF (YRECLDP%LAERLIQCOLL) THEN
              ! 5.0 = N**0.333 with N=125 cm-3
              ZCFPR = ZCFPR*(YRECLDP%RCCN / PCCN(JL, JK))**0.333_JPRB
            END IF

            ZZCO = ZZCO*ZCFPR
            ZLCRIT = ZLCRIT / MAX(ZCFPR, ZEPSEC)

            IF (ZLIQCLD / ZLCRIT < 20.0_JPRB) THEN
              ! Security for exp for some compilers
              ZRAINAUT = ZZCO*(1.0_JPRB - EXP(-(ZLIQCLD / ZLCRIT)**2))
            ELSE
              ZRAINAUT = ZZCO
            END IF

            ! rain freezes instantly
            IF (ZTP1(JL, JK) <= RTT) THEN
              ZSOLQB(NCLDQS, NCLDQL) = ZSOLQB(NCLDQS, NCLDQL) + ZRAINAUT
            ELSE
              ZSOLQB(NCLDQR, NCLDQL) = ZSOLQB(NCLDQR, NCLDQL) + ZRAINAUT
            END IF

            !--------------------------------------------------------
            !-
            !- Warm-rain process follow Khairoutdinov and Kogan (2000)
            !-
            !--------------------------------------------------------
          ELSE IF (IWARMRAIN == 2) THEN

            IF (PLSM(JL) > 0.5_JPRB) THEN
              ! land
              ZCONST = YRECLDP%RCL_KK_CLOUD_NUM_LAND
              ZLCRIT = YRECLDP%RCLCRIT_LAND
            ELSE
              ! ocean
              ZCONST = YRECLDP%RCL_KK_CLOUD_NUM_SEA
              ZLCRIT = YRECLDP%RCLCRIT_SEA
            END IF

            IF (ZLIQCLD > ZLCRIT) THEN

              ZRAINAUT = 1.5_JPRB*ZA(JL, JK)*PTSPHY*YRECLDP%RCL_KKAAU*ZLIQCLD**YRECLDP%RCL_KKBAUQ*ZCONST**YRECLDP%RCL_KKBAUN

              ZRAINAUT = MIN(ZRAINAUT, ZQXFG(NCLDQL))
              IF (ZRAINAUT < ZEPSEC)               ZRAINAUT = 0.0_JPRB

              ZRAINACC = 2.0_JPRB*ZA(JL, JK)*PTSPHY*YRECLDP%RCL_KKAAC*(ZLIQCLD*ZRAINCLD)**YRECLDP%RCL_KKBAC

              ZRAINACC = MIN(ZRAINACC, ZQXFG(NCLDQL))
              IF (ZRAINACC < ZEPSEC)               ZRAINACC = 0.0_JPRB

            ELSE
              ZRAINAUT = 0.0_JPRB
              ZRAINACC = 0.0_JPRB
            END IF

            ! If temperature < 0, then autoconversion produces snow rather than rain
            ! Explicit
            IF (ZTP1(JL, JK) <= RTT) THEN
              ZSOLQA(NCLDQS, NCLDQL) = ZSOLQA(NCLDQS, NCLDQL) + ZRAINAUT
              ZSOLQA(NCLDQS, NCLDQL) = ZSOLQA(NCLDQS, NCLDQL) + ZRAINACC
              ZSOLQA(NCLDQL, NCLDQS) = ZSOLQA(NCLDQL, NCLDQS) - ZRAINAUT
              ZSOLQA(NCLDQL, NCLDQS) = ZSOLQA(NCLDQL, NCLDQS) - ZRAINACC
            ELSE
              ZSOLQA(NCLDQR, NCLDQL) = ZSOLQA(NCLDQR, NCLDQL) + ZRAINAUT
              ZSOLQA(NCLDQR, NCLDQL) = ZSOLQA(NCLDQR, NCLDQL) + ZRAINACC
              ZSOLQA(NCLDQL, NCLDQR) = ZSOLQA(NCLDQL, NCLDQR) - ZRAINAUT
              ZSOLQA(NCLDQL, NCLDQR) = ZSOLQA(NCLDQL, NCLDQR) - ZRAINACC
            END IF

          END IF
          ! on IWARMRAIN

        END IF
        ! on ZLIQCLD > ZEPSEC


        !----------------------------------------------------------------------
        ! RIMING - COLLECTION OF CLOUD LIQUID DROPS BY SNOW AND ICE
        !      only active if T<0degC and supercooled liquid water is present
        !      AND if not Sundquist autoconversion (as this includes riming)
        !----------------------------------------------------------------------
        IF (IWARMRAIN > 1) THEN

          IF (ZTP1(JL, JK) <= RTT .and. ZLIQCLD > ZEPSEC) THEN

            ! Fallspeed air density correction
            ZFALLCORR = (YRECLDP%RDENSREF / ZRHO)**0.4_JPRB

            !------------------------------------------------------------------
            ! Riming of snow by cloud water - implicit in lwc
            !------------------------------------------------------------------
            IF (ZSNOWCLD > ZEPSEC .and. ZCOVPTOT > 0.01_JPRB) THEN

              ! Calculate riming term
              ! Factor of liq water taken out because implicit
              ZSNOWRIME =  &
              & 0.3_JPRB*ZCOVPTOT*PTSPHY*YRECLDP%RCL_CONST7S*ZFALLCORR*(ZRHO*ZSNOWCLD*YRECLDP%RCL_CONST1S)**YRECLDP%RCL_CONST8S

              ! Limit snow riming term
              ZSNOWRIME = MIN(ZSNOWRIME, 1.0_JPRB)

              ZSOLQB(NCLDQS, NCLDQL) = ZSOLQB(NCLDQS, NCLDQL) + ZSNOWRIME

            END IF

            !------------------------------------------------------------------
            ! Riming of ice by cloud water - implicit in lwc
            ! NOT YET ACTIVE
            !------------------------------------------------------------------
            !      IF (ZICECLD(JL)>ZEPSEC .AND. ZA(JL,JK)>0.01_JPRB) THEN
            !
            !        ! Calculate riming term
            !        ! Factor of liq water taken out because implicit
            !        ZSNOWRIME(JL) = ZA(JL,JK)*PTSPHY*RCL_CONST7S*ZFALLCORR &
            !     &                  *(ZRHO(JL)*ZICECLD(JL)*RCL_CONST1S)**RCL_CONST8S
            !
            !        ! Limit ice riming term
            !        ZSNOWRIME(JL)=MIN(ZSNOWRIME(JL),1.0_JPRB)
            !
            !        ZSOLQB(JL,NCLDQI,NCLDQL) = ZSOLQB(JL,NCLDQI,NCLDQL) + ZSNOWRIME(JL)
            !
            !      ENDIF
          END IF

        END IF
        ! on IWARMRAIN > 1


        !----------------------------------------------------------------------
        ! 4.4a  MELTING OF SNOW and ICE
        !       with new implicit solver this also has to treat snow or ice
        !       precipitating from the level above... i.e. local ice AND flux.
        !       in situ ice and snow: could arise from LS advection or warming
        !       falling ice and snow: arrives by precipitation process
        !----------------------------------------------------------------------

        ZICETOT = ZQXFG(NCLDQI) + ZQXFG(NCLDQS)
        ZMELTMAX = 0.0_JPRB

        ! If there are frozen hydrometeors present and dry-bulb temperature > 0degC
        IF (ZICETOT > ZEPSEC .and. ZTP1(JL, JK) > RTT) THEN

          ! Calculate subsaturation
          ZSUBSAT = MAX(ZQSICE(JL, JK) - ZQX(JL, JK, NCLDQV), 0.0_JPRB)

          ! Calculate difference between dry-bulb (ZTP1) and the temperature
          ! at which the wet-bulb=0degC (RTT-ZSUBSAT*....) using an approx.
          ! Melting only occurs if the wet-bulb temperature >0
          ! i.e. warming of ice particle due to melting > cooling
          ! due to evaporation.
          ZTDMTW0 = ZTP1(JL, JK) - RTT - ZSUBSAT*(ZTW1 + ZTW2*(PAP(JL, JK) - ZTW3) - ZTW4*(ZTP1(JL, JK) - ZTW5))
          ! Not implicit yet...
          ! Ensure ZCONS1 is positive so that ZMELTMAX=0 if ZTDMTW0<0
          ZCONS1 = ABS((PTSPHY*(1.0_JPRB + 0.5_JPRB*ZTDMTW0)) / YRECLDP%RTAUMEL)
          ZMELTMAX = MAX(ZTDMTW0*ZCONS1*ZRLDCP, 0.0_JPRB)
        END IF

        ! Loop over frozen hydrometeors (ice, snow)
        DO JM=1,NCLV
          IF (IPHASE(JM) == 2) THEN
            JN = IMELT(JM)
            IF (ZMELTMAX > ZEPSEC .and. ZICETOT > ZEPSEC) THEN
              ! Apply melting in same proportion as frozen hydrometeor fractions
              ZALFA = ZQXFG(JM) / ZICETOT
              ZMELT = MIN(ZQXFG(JM), ZALFA*ZMELTMAX)
              ! needed in first guess
              ! This implies that zqpretot has to be recalculated below
              ! since is not conserved here if ice falls and liquid doesn't
              ZQXFG(JM) = ZQXFG(JM) - ZMELT
              ZQXFG(JN) = ZQXFG(JN) + ZMELT
              ZSOLQA(JN, JM) = ZSOLQA(JN, JM) + ZMELT
              ZSOLQA(JM, JN) = ZSOLQA(JM, JN) - ZMELT
            END IF
          END IF
        END DO

        !----------------------------------------------------------------------
        ! 4.4b  FREEZING of RAIN
        !----------------------------------------------------------------------

        ! If rain present
        IF (ZQX(JL, JK, NCLDQR) > ZEPSEC) THEN

          IF (ZTP1(JL, JK) <= RTT .and. ZTP1(JL, JK - 1) > RTT) THEN
            ! Base of melting layer/top of refreezing layer so
            ! store rain/snow fraction for precip type diagnosis
            ! If mostly rain, then supercooled rain slow to freeze
            ! otherwise faster to freeze (snow or ice pellets)
            ZQPRETOT = MAX(ZQX(JL, JK, NCLDQS) + ZQX(JL, JK, NCLDQR), ZEPSEC)
            PRAINFRAC_TOPRFZ(JL) = ZQX(JL, JK, NCLDQR) / ZQPRETOT
            IF (PRAINFRAC_TOPRFZ(JL) > 0.8) THEN
              LLRAINLIQ = .true.
            ELSE
              LLRAINLIQ = .false.
            END IF
          END IF

          ! If temperature less than zero
          IF (ZTP1(JL, JK) < RTT) THEN

            IF (PRAINFRAC_TOPRFZ(JL) > 0.8) THEN

              ! Majority of raindrops completely melted
              ! Refreezing is by slow heterogeneous freezing

              ! Slope of rain particle size distribution
              ZLAMBDA = (YRECLDP%RCL_FAC1 / ((ZRHO*ZQX(JL, JK, NCLDQR))))**YRECLDP%RCL_FAC2

              ! Calculate freezing rate based on Bigg(1953) and Wisner(1972)
              ZTEMP = YRECLDP%RCL_FZRAB*(ZTP1(JL, JK) - RTT)
              ZFRZ = PTSPHY*(YRECLDP%RCL_CONST5R / ZRHO)*(EXP(ZTEMP) - 1._JPRB)*ZLAMBDA**YRECLDP%RCL_CONST6R
              ZFRZMAX = MAX(ZFRZ, 0.0_JPRB)

            ELSE

              ! Majority of raindrops only partially melted
              ! Refreeze with a shorter timescale (reverse of melting...for now)

              ZCONS1 = ABS((PTSPHY*(1.0_JPRB + 0.5_JPRB*(RTT - ZTP1(JL, JK)))) / YRECLDP%RTAUMEL)
              ZFRZMAX = MAX((RTT - ZTP1(JL, JK))*ZCONS1*ZRLDCP, 0.0_JPRB)

            END IF

            IF (ZFRZMAX > ZEPSEC) THEN
              ZFRZ = MIN(ZQX(JL, JK, NCLDQR), ZFRZMAX)
              ZSOLQA(NCLDQS, NCLDQR) = ZSOLQA(NCLDQS, NCLDQR) + ZFRZ
              ZSOLQA(NCLDQR, NCLDQS) = ZSOLQA(NCLDQR, NCLDQS) - ZFRZ
            END IF
          END IF

        END IF


        !----------------------------------------------------------------------
        ! 4.4c  FREEZING of LIQUID
        !----------------------------------------------------------------------
        ! not implicit yet...
        ZFRZMAX = MAX((YRECLDP%RTHOMO - ZTP1(JL, JK))*ZRLDCP, 0.0_JPRB)

        JM = NCLDQL
        JN = IMELT(JM)
        IF (ZFRZMAX > ZEPSEC .and. ZQXFG(JM) > ZEPSEC) THEN
          ZFRZ = MIN(ZQXFG(JM), ZFRZMAX)
          ZSOLQA(JN, JM) = ZSOLQA(JN, JM) + ZFRZ
          ZSOLQA(JM, JN) = ZSOLQA(JM, JN) - ZFRZ
        END IF

        !----------------------------------------------------------------------
        ! 4.5   EVAPORATION OF RAIN/SNOW
        !----------------------------------------------------------------------

        !----------------------------------------
        ! Rain evaporation scheme from Sundquist
        !----------------------------------------
        IF (IEVAPRAIN == 1) THEN

          ! Rain


          ZZRH = YRECLDP%RPRECRHMAX + ((1.0_JPRB - YRECLDP%RPRECRHMAX)*ZCOVPMAX) / MAX(ZEPSEC, 1.0_JPRB - ZA(JL, JK))
          ZZRH = MIN(MAX(ZZRH, YRECLDP%RPRECRHMAX), 1.0_JPRB)

          ZQE = (ZQX(JL, JK, NCLDQV) - ZA(JL, JK)*ZQSLIQ(JL, JK)) / MAX(ZEPSEC, 1.0_JPRB - ZA(JL, JK))
          !---------------------------------------------
          ! humidity in moistest ZCOVPCLR part of domain
          !---------------------------------------------
          ZQE = MAX(0.0_JPRB, MIN(ZQE, ZQSLIQ(JL, JK)))
          LLO1 = ZCOVPCLR > ZEPSEC .and. ZQXFG(NCLDQR) > ZEPSEC .and. ZQE < ZZRH*ZQSLIQ(JL, JK)

          IF (LLO1) THEN
            ! note: zpreclr is a rain flux
            ZPRECLR = (ZQXFG(NCLDQR)*ZCOVPCLR) / SIGN(MAX(ABS(ZCOVPTOT*ZDTGDP), ZEPSILON), ZCOVPTOT*ZDTGDP)

            !--------------------------------------
            ! actual microphysics formula in zbeta
            !--------------------------------------

            ZBETA1 = ((SQRT(PAP(JL, JK) / PAPH(JL, KLEV + 1)) / YRECLDP%RVRFACTOR)*ZPRECLR) / MAX(ZCOVPCLR, ZEPSEC)

            ZBETA = RG*YRECLDP%RPECONS*0.5_JPRB*ZBETA1**0.5777_JPRB

            ZDENOM = 1.0_JPRB + ZBETA*PTSPHY*ZCORQSLIQ
            ZDPR = ((ZCOVPCLR*ZBETA*(ZQSLIQ(JL, JK) - ZQE)) / ZDENOM)*ZDP*ZRG_R
            ZDPEVAP = ZDPR*ZDTGDP

            !---------------------------------------------------------
            ! add evaporation term to explicit sink.
            ! this has to be explicit since if treated in the implicit
            ! term evaporation can not reduce rain to zero and model
            ! produces small amounts of rainfall everywhere.
            !---------------------------------------------------------

            ! Evaporate rain
            ZEVAP = MIN(ZDPEVAP, ZQXFG(NCLDQR))

            ZSOLQA(NCLDQV, NCLDQR) = ZSOLQA(NCLDQV, NCLDQR) + ZEVAP
            ZSOLQA(NCLDQR, NCLDQV) = ZSOLQA(NCLDQR, NCLDQV) - ZEVAP

            !-------------------------------------------------------------
            ! Reduce the total precip coverage proportional to evaporation
            ! to mimic the previous scheme which had a diagnostic
            ! 2-flux treatment, abandoned due to the new prognostic precip
            !-------------------------------------------------------------
            ZCOVPTOT = MAX(YRECLDP%RCOVPMIN, ZCOVPTOT - MAX(0.0_JPRB, ((ZCOVPTOT - ZA(JL, JK))*ZEVAP) / ZQXFG(NCLDQR)))

            ! Update fg field
            ZQXFG(NCLDQR) = ZQXFG(NCLDQR) - ZEVAP

          END IF


          !---------------------------------------------------------
          ! Rain evaporation scheme based on Abel and Boutle (2013)
          !---------------------------------------------------------
        ELSE IF (IEVAPRAIN == 2) THEN


          !-----------------------------------------------------------------------
          ! Calculate relative humidity limit for rain evaporation
          ! to avoid cloud formation and saturation of the grid box
          !-----------------------------------------------------------------------
          ! Limit RH for rain evaporation dependent on precipitation fraction
          ZZRH = YRECLDP%RPRECRHMAX + ((1.0_JPRB - YRECLDP%RPRECRHMAX)*ZCOVPMAX) / MAX(ZEPSEC, 1.0_JPRB - ZA(JL, JK))
          ZZRH = MIN(MAX(ZZRH, YRECLDP%RPRECRHMAX), 1.0_JPRB)

          ! Critical relative humidity
          !ZRHC=RAMID
          !ZSIGK=PAP(JL,JK)/PAPH(JL,KLEV+1)
          ! Increase RHcrit to 1.0 towards the surface (eta>0.8)
          !IF(ZSIGK > 0.8_JPRB) THEN
          !  ZRHC=RAMID+(1.0_JPRB-RAMID)*((ZSIGK-0.8_JPRB)/0.2_JPRB)**2
          !ENDIF
          !ZZRH = MIN(ZRHC,ZZRH)

          ! Further limit RH for rain evaporation to 80% (RHcrit in free troposphere)
          ZZRH = MIN(0.8_JPRB, ZZRH)

          ZQE = MAX(0.0_JPRB, MIN(ZQX(JL, JK, NCLDQV), ZQSLIQ(JL, JK)))

          LLO1 = ZCOVPCLR > ZEPSEC .and. ZQXFG(NCLDQR) > ZEPSEC .and. ZQE < ZZRH*ZQSLIQ(JL, JK)

          IF (LLO1) THEN

            !-------------------------------------------
            ! Abel and Boutle (2012) evaporation
            !-------------------------------------------
            ! Calculate local precipitation (kg/kg)
            ZPRECLR = ZQXFG(NCLDQR) / ZCOVPTOT

            ! Fallspeed air density correction
            ZFALLCORR = (YRECLDP%RDENSREF / ZRHO)**0.4

            ! Saturation vapour pressure with respect to liquid phase
            ZESATLIQ = (RV / RD)*FOEELIQ(ZTP1(JL, JK))

            ! Slope of particle size distribution
            ZLAMBDA = (YRECLDP%RCL_FAC1 / ((ZRHO*ZPRECLR)))**YRECLDP%RCL_FAC2              ! ZPRECLR=kg/kg

            ZEVAP_DENOM = YRECLDP%RCL_CDENOM1*ZESATLIQ - YRECLDP%RCL_CDENOM2*ZTP1(JL, JK)*ZESATLIQ +  &
            & YRECLDP%RCL_CDENOM3*ZTP1(JL, JK)**3._JPRB*PAP(JL, JK)

            ! Temperature dependent conductivity
            ZCORR2 = ((ZTP1(JL, JK) / 273._JPRB)**1.5_JPRB*393._JPRB) / (ZTP1(JL, JK) + 120._JPRB)
            ZKA = YRECLDP%RCL_KA273*ZCORR2

            ZSUBSAT = MAX(ZZRH*ZQSLIQ(JL, JK) - ZQE, 0.0_JPRB)

            ZBETA = (0.5_JPRB / ZQSLIQ(JL, JK))*ZTP1(JL, JK)**2._JPRB*ZESATLIQ*YRECLDP%RCL_CONST1R*(ZCORR2 /  &
            & ZEVAP_DENOM)*(0.78_JPRB / (ZLAMBDA**YRECLDP%RCL_CONST4R) + (YRECLDP%RCL_CONST2R*(ZRHO*ZFALLCORR)**0.5_JPRB) /  &
            & ((ZCORR2**0.5_JPRB*ZLAMBDA**YRECLDP%RCL_CONST3R)))

            ZDENOM = 1.0_JPRB + ZBETA*PTSPHY              !*ZCORQSLIQ(JL)
            ZDPEVAP = (ZCOVPCLR*ZBETA*PTSPHY*ZSUBSAT) / ZDENOM

            !---------------------------------------------------------
            ! Add evaporation term to explicit sink.
            ! this has to be explicit since if treated in the implicit
            ! term evaporation can not reduce rain to zero and model
            ! produces small amounts of rainfall everywhere.
            !---------------------------------------------------------

            ! Limit rain evaporation
            ZEVAP = MIN(ZDPEVAP, ZQXFG(NCLDQR))

            ZSOLQA(NCLDQV, NCLDQR) = ZSOLQA(NCLDQV, NCLDQR) + ZEVAP
            ZSOLQA(NCLDQR, NCLDQV) = ZSOLQA(NCLDQR, NCLDQV) - ZEVAP

            !-------------------------------------------------------------
            ! Reduce the total precip coverage proportional to evaporation
            ! to mimic the previous scheme which had a diagnostic
            ! 2-flux treatment, abandoned due to the new prognostic precip
            !-------------------------------------------------------------
            ZCOVPTOT = MAX(YRECLDP%RCOVPMIN, ZCOVPTOT - MAX(0.0_JPRB, ((ZCOVPTOT - ZA(JL, JK))*ZEVAP) / ZQXFG(NCLDQR)))

            ! Update fg field
            ZQXFG(NCLDQR) = ZQXFG(NCLDQR) - ZEVAP

          END IF

        END IF
        ! on IEVAPRAIN

        !----------------------------------------------------------------------
        ! 4.5   EVAPORATION OF SNOW
        !----------------------------------------------------------------------
        ! Snow
        IF (IEVAPSNOW == 1) THEN

          ZZRH = YRECLDP%RPRECRHMAX + ((1.0_JPRB - YRECLDP%RPRECRHMAX)*ZCOVPMAX) / MAX(ZEPSEC, 1.0_JPRB - ZA(JL, JK))
          ZZRH = MIN(MAX(ZZRH, YRECLDP%RPRECRHMAX), 1.0_JPRB)
          ZQE = (ZQX(JL, JK, NCLDQV) - ZA(JL, JK)*ZQSICE(JL, JK)) / MAX(ZEPSEC, 1.0_JPRB - ZA(JL, JK))

          !---------------------------------------------
          ! humidity in moistest ZCOVPCLR part of domain
          !---------------------------------------------
          ZQE = MAX(0.0_JPRB, MIN(ZQE, ZQSICE(JL, JK)))
          LLO1 = ZCOVPCLR > ZEPSEC .and. ZQXFG(NCLDQS) > ZEPSEC .and. ZQE < ZZRH*ZQSICE(JL, JK)

          IF (LLO1) THEN
            ! note: zpreclr is a rain flux a
            ZPRECLR = (ZQXFG(NCLDQS)*ZCOVPCLR) / SIGN(MAX(ABS(ZCOVPTOT*ZDTGDP), ZEPSILON), ZCOVPTOT*ZDTGDP)

            !--------------------------------------
            ! actual microphysics formula in zbeta
            !--------------------------------------

            ZBETA1 = ((SQRT(PAP(JL, JK) / PAPH(JL, KLEV + 1)) / YRECLDP%RVRFACTOR)*ZPRECLR) / MAX(ZCOVPCLR, ZEPSEC)

            ZBETA = RG*YRECLDP%RPECONS*ZBETA1**0.5777_JPRB

            ZDENOM = 1.0_JPRB + ZBETA*PTSPHY*ZCORQSICE
            ZDPR = ((ZCOVPCLR*ZBETA*(ZQSICE(JL, JK) - ZQE)) / ZDENOM)*ZDP*ZRG_R
            ZDPEVAP = ZDPR*ZDTGDP

            !---------------------------------------------------------
            ! add evaporation term to explicit sink.
            ! this has to be explicit since if treated in the implicit
            ! term evaporation can not reduce snow to zero and model
            ! produces small amounts of snowfall everywhere.
            !---------------------------------------------------------

            ! Evaporate snow
            ZEVAP = MIN(ZDPEVAP, ZQXFG(NCLDQS))

            ZSOLQA(NCLDQV, NCLDQS) = ZSOLQA(NCLDQV, NCLDQS) + ZEVAP
            ZSOLQA(NCLDQS, NCLDQV) = ZSOLQA(NCLDQS, NCLDQV) - ZEVAP

            !-------------------------------------------------------------
            ! Reduce the total precip coverage proportional to evaporation
            ! to mimic the previous scheme which had a diagnostic
            ! 2-flux treatment, abandoned due to the new prognostic precip
            !-------------------------------------------------------------
            ZCOVPTOT = MAX(YRECLDP%RCOVPMIN, ZCOVPTOT - MAX(0.0_JPRB, ((ZCOVPTOT - ZA(JL, JK))*ZEVAP) / ZQXFG(NCLDQS)))

            !Update first guess field
            ZQXFG(NCLDQS) = ZQXFG(NCLDQS) - ZEVAP

          END IF
          !---------------------------------------------------------
        ELSE IF (IEVAPSNOW == 2) THEN



          !-----------------------------------------------------------------------
          ! Calculate relative humidity limit for snow evaporation
          !-----------------------------------------------------------------------
          ZZRH = YRECLDP%RPRECRHMAX + ((1.0_JPRB - YRECLDP%RPRECRHMAX)*ZCOVPMAX) / MAX(ZEPSEC, 1.0_JPRB - ZA(JL, JK))
          ZZRH = MIN(MAX(ZZRH, YRECLDP%RPRECRHMAX), 1.0_JPRB)
          ZQE = (ZQX(JL, JK, NCLDQV) - ZA(JL, JK)*ZQSICE(JL, JK)) / MAX(ZEPSEC, 1.0_JPRB - ZA(JL, JK))

          !---------------------------------------------
          ! humidity in moistest ZCOVPCLR part of domain
          !---------------------------------------------
          ZQE = MAX(0.0_JPRB, MIN(ZQE, ZQSICE(JL, JK)))
          LLO1 = ZCOVPCLR > ZEPSEC .and. ZQX(JL, JK, NCLDQS) > ZEPSEC .and. ZQE < ZZRH*ZQSICE(JL, JK)

          IF (LLO1) THEN

            ! Calculate local precipitation (kg/kg)
            ZPRECLR = ZQX(JL, JK, NCLDQS) / ZCOVPTOT
            ZVPICE = (FOEEICE(ZTP1(JL, JK))*RV) / RD

            ! Particle size distribution
            ! ZTCG increases Ni with colder temperatures - essentially a
            ! Fletcher or Meyers scheme?
            ZTCG = 1.0_JPRB              !v1 EXP(RCL_X3I*(273.15_JPRB-ZTP1(JL,JK))/8.18_JPRB)
            ! ZFACX1I modification is based on Andrew Barrett's results
            ZFACX1S = 1.0_JPRB              !v1 (ZICE0/1.E-5_JPRB)**0.627_JPRB

            ZAPLUSB =  &
            & YRECLDP%RCL_APB1*ZVPICE - YRECLDP%RCL_APB2*ZVPICE*ZTP1(JL, JK) + PAP(JL, JK)*YRECLDP%RCL_APB3*ZTP1(JL, JK)**3
            ZCORRFAC = (1.0 / ZRHO)**0.5
            ZCORRFAC2 = ((ZTP1(JL, JK) / 273.0)**1.5)*(393.0 / (ZTP1(JL, JK) + 120.0))

            ZPR02 = (ZRHO*ZPRECLR*YRECLDP%RCL_CONST1S) / ((ZTCG*ZFACX1S))

            ZTERM1 = ((ZQSICE(JL, JK) - ZQE)*ZTP1(JL, JK)**2*ZVPICE*ZCORRFAC2*ZTCG*YRECLDP%RCL_CONST2S*ZFACX1S) /  &
            & ((ZRHO*ZAPLUSB*ZQSICE(JL, JK)))
            ZTERM2 = 0.65*YRECLDP%RCL_CONST6S*ZPR02**YRECLDP%RCL_CONST4S +  &
            & (YRECLDP%RCL_CONST3S*ZCORRFAC**0.5*ZRHO**0.5*ZPR02**YRECLDP%RCL_CONST5S) / ZCORRFAC2**0.5

            ZDPEVAP = MAX(ZCOVPCLR*ZTERM1*ZTERM2*PTSPHY, 0.0_JPRB)

            !--------------------------------------------------------------------
            ! Limit evaporation to snow amount
            !--------------------------------------------------------------------
            ZEVAP = MIN(ZDPEVAP, ZEVAPLIMICE)
            ZEVAP = MIN(ZEVAP, ZQX(JL, JK, NCLDQS))


            ZSOLQA(NCLDQV, NCLDQS) = ZSOLQA(NCLDQV, NCLDQS) + ZEVAP
            ZSOLQA(NCLDQS, NCLDQV) = ZSOLQA(NCLDQS, NCLDQV) - ZEVAP

            !-------------------------------------------------------------
            ! Reduce the total precip coverage proportional to evaporation
            ! to mimic the previous scheme which had a diagnostic
            ! 2-flux treatment, abandoned due to the new prognostic precip
            !-------------------------------------------------------------
            ZCOVPTOT = MAX(YRECLDP%RCOVPMIN, ZCOVPTOT - MAX(0.0_JPRB, ((ZCOVPTOT - ZA(JL, JK))*ZEVAP) / ZQX(JL, JK, NCLDQS)))

            !Update first guess field
            ZQXFG(NCLDQS) = ZQXFG(NCLDQS) - ZEVAP

          END IF

        END IF
        ! on IEVAPSNOW

        !--------------------------------------
        ! Evaporate small precipitation amounts
        !--------------------------------------
        DO JM=1,NCLV
          IF (LLFALL(JM)) THEN
            IF (ZQXFG(JM) < YRECLDP%RLMIN) THEN
              ZSOLQA(NCLDQV, JM) = ZSOLQA(NCLDQV, JM) + ZQXFG(JM)
              ZSOLQA(JM, NCLDQV) = ZSOLQA(JM, NCLDQV) - ZQXFG(JM)
            END IF
          END IF
        END DO

        !######################################################################
        !            5.0  *** SOLVERS FOR A AND L ***
        ! now use an implicit solution rather than exact solution
        ! solver is forward in time, upstream difference for advection
        !######################################################################

        !---------------------------
        ! 5.1 solver for cloud cover
        !---------------------------
        ZANEW = (ZA(JL, JK) + ZSOLAC) / (1.0_JPRB + ZSOLAB)
        ZANEW = MIN(ZANEW, 1.0_JPRB)
        IF (ZANEW < YRECLDP%RAMIN)         ZANEW = 0.0_JPRB
        ZDA = ZANEW - ZAORIG(JL, JK)
        !---------------------------------
        ! variables needed for next level
        !---------------------------------
        ZANEWM1 = ZANEW

        !--------------------------------
        ! 5.2 solver for the microphysics
        !--------------------------------

        !--------------------------------------------------------------
        ! Truncate explicit sinks to avoid negatives
        ! Note: Species are treated in the order in which they run out
        ! since the clipping will alter the balance for the other vars
        !--------------------------------------------------------------

        DO JM=1,NCLV
!$claw nodep
          DO JN=1,NCLV
            LLINDEX3(JN, JM) = .false.
          END DO
          ZSINKSUM(JM) = 0.0_JPRB
        END DO

        !----------------------------
        ! collect sink terms and mark
        !----------------------------
        DO JM=1,NCLV
          DO JN=1,NCLV
            ZSINKSUM(JM) = ZSINKSUM(JM) - ZSOLQA(JM, JN)              ! +ve total is bad
          END DO
        END DO

        !---------------------------------------
        ! calculate overshoot and scaling factor
        !---------------------------------------
        DO JM=1,NCLV
          ZMAX = MAX(ZQX(JL, JK, JM), ZEPSEC)
          ZRAT = MAX(ZSINKSUM(JM), ZMAX)
          ZRATIO(JM) = ZMAX / ZRAT
        END DO

        !--------------------------------------------
        ! scale the sink terms, in the correct order,
        ! recalculating the scale factor each time
        !--------------------------------------------
        DO JM=1,NCLV
          ZSINKSUM(JM) = 0.0_JPRB
        END DO

        !----------------
        ! recalculate sum
        !----------------
        DO JM=1,NCLV
          PSUM_SOLQA = 0.0
          DO JN=1,NCLV
            PSUM_SOLQA = PSUM_SOLQA + ZSOLQA(JM, JN)
          END DO
          ! ZSINKSUM(JL,JM)=ZSINKSUM(JL,JM)-SUM(ZSOLQA(JL,JM,1:NCLV))
          ZSINKSUM(JM) = ZSINKSUM(JM) - PSUM_SOLQA
          !---------------------------
          ! recalculate scaling factor
          !---------------------------
          ZMM = MAX(ZQX(JL, JK, JM), ZEPSEC)
          ZRR = MAX(ZSINKSUM(JM), ZMM)
          ZRATIO(JM) = ZMM / ZRR
          !------
          ! scale
          !------
          ZZRATIO = ZRATIO(JM)
          !DIR$ IVDEP
          !DIR$ PREFERVECTOR
          DO JN=1,NCLV
            IF (ZSOLQA(JM, JN) < 0.0_JPRB) THEN
              ZSOLQA(JM, JN) = ZSOLQA(JM, JN)*ZZRATIO
              ZSOLQA(JN, JM) = ZSOLQA(JN, JM)*ZZRATIO
            END IF
          END DO
        END DO

        !--------------------------------------------------------------
        ! 5.2.2 Solver
        !------------------------

        !------------------------
        ! set the LHS of equation
        !------------------------
        DO JM=1,NCLV
          DO JN=1,NCLV
            !----------------------------------------------
            ! diagonals: microphysical sink terms+transport
            !----------------------------------------------
            IF (JN == JM) THEN
              ZQLHS(JN, JM) = 1.0_JPRB + ZFALLSINK(JM)
              DO JO=1,NCLV
                ZQLHS(JN, JM) = ZQLHS(JN, JM) + ZSOLQB(JO, JN)
              END DO
              !------------------------------------------
              ! non-diagonals: microphysical source terms
              !------------------------------------------
            ELSE
              ZQLHS(JN, JM) = -ZSOLQB(JN, JM)                ! here is the delta T - missing from doc.
            END IF
          END DO
        END DO

        !------------------------
        ! set the RHS of equation
        !------------------------
        DO JM=1,NCLV
          !---------------------------------
          ! sum the explicit source and sink
          !---------------------------------
          ZEXPLICIT = 0.0_JPRB
          DO JN=1,NCLV
            ZEXPLICIT = ZEXPLICIT + ZSOLQA(JM, JN)              ! sum over middle index
          END DO
          ZQXN(JM) = ZQX(JL, JK, JM) + ZEXPLICIT
        END DO

        !-----------------------------------
        ! *** solve by LU decomposition: ***
        !-----------------------------------

        ! Note: This fast way of solving NCLVxNCLV system
        !       assumes a good behaviour (i.e. non-zero diagonal
        !       terms with comparable orders) of the matrix stored
        !       in ZQLHS. For the moment this is the case but
        !       be aware to preserve it when doing eventual
        !       modifications.

        ! Non pivoting recursive factorization
        DO JN=1,NCLV - 1
          ! number of steps
          DO JM=JN + 1,NCLV
            ! row index
            ZQLHS(JM, JN) = ZQLHS(JM, JN) / ZQLHS(JN, JN)
            DO IK=JN + 1,NCLV
              ! column index
              ZQLHS(JM, IK) = ZQLHS(JM, IK) - ZQLHS(JM, JN)*ZQLHS(JN, IK)
            END DO
          END DO
        END DO

        ! Backsubstitution
        !  step 1
        DO JN=2,NCLV
          DO JM=1,JN - 1
            ZQXN(JN) = ZQXN(JN) - ZQLHS(JN, JM)*ZQXN(JM)
          END DO
        END DO
        !  step 2
        ZQXN(NCLV) = ZQXN(NCLV) / ZQLHS(NCLV, NCLV)
        DO JN=NCLV - 1,1,-1
          DO JM=JN + 1,NCLV
            ZQXN(JN) = ZQXN(JN) - ZQLHS(JN, JM)*ZQXN(JM)
          END DO
          ZQXN(JN) = ZQXN(JN) / ZQLHS(JN, JN)
        END DO

        ! Ensure no small values (including negatives) remain in cloud variables nor
        ! precipitation rates.
        ! Evaporate l,i,r,s to water vapour. Latent heating taken into account below
        DO JN=1,NCLV - 1
          IF (ZQXN(JN) < ZEPSEC) THEN
            ZQXN(NCLDQV) = ZQXN(NCLDQV) + ZQXN(JN)
            ZQXN(JN) = 0.0_JPRB
          END IF
        END DO

        !--------------------------------
        ! variables needed for next level
        !--------------------------------
        DO JM=1,NCLV
          ZQXNM1(JM) = ZQXN(JM)
          ZQXN2D(JL, JK, JM) = ZQXN(JM)
        END DO

        !------------------------------------------------------------------------
        ! 5.3 Precipitation/sedimentation fluxes to next level
        !     diagnostic precipitation fluxes
        !     It is this scaled flux that must be used for source to next layer
        !------------------------------------------------------------------------

        DO JM=1,NCLV
          ZPFPLSX(JL, JK + 1, JM) = ZFALLSINK(JM)*ZQXN(JM)*ZRDTGDP
        END DO

        ! Ensure precipitation fraction is zero if no precipitation
        ZQPRETOT = ZPFPLSX(JL, JK + 1, NCLDQS) + ZPFPLSX(JL, JK + 1, NCLDQR)
        IF (ZQPRETOT < ZEPSEC) THEN
          ZCOVPTOT = 0.0_JPRB
        END IF

        !######################################################################
        !              6  *** UPDATE TENDANCIES ***
        !######################################################################

        !--------------------------------
        ! 6.1 Temperature and CLV budgets
        !--------------------------------

        DO JM=1,NCLV - 1

          ! calculate fluxes in and out of box for conservation of TL
          ZFLUXQ(JM) = ZPSUPSATSRCE(JM) + ZCONVSRCE(JM) + ZFALLSRCE(JM) - (ZFALLSINK(JM) + ZCONVSINK(JM))*ZQXN(JM)

          IF (IPHASE(JM) == 1) THEN
            TENDENCY_LOC_T(JL, JK) = TENDENCY_LOC_T(JL, JK) + RALVDCP*(ZQXN(JM) - ZQX(JL, JK, JM) - ZFLUXQ(JM))*ZQTMST
          END IF

          IF (IPHASE(JM) == 2) THEN
            TENDENCY_LOC_T(JL, JK) = TENDENCY_LOC_T(JL, JK) + RALSDCP*(ZQXN(JM) - ZQX(JL, JK, JM) - ZFLUXQ(JM))*ZQTMST
          END IF

          !----------------------------------------------------------------------
          ! New prognostic tendencies - ice,liquid rain,snow
          ! Note: CLV arrays use PCLV in calculation of tendency while humidity
          !       uses ZQX. This is due to clipping at start of cloudsc which
          !       include the tendency already in TENDENCY_LOC_T and TENDENCY_LOC_q. ZQX was reset
          !----------------------------------------------------------------------
          TENDENCY_LOC_CLD(JL, JK, JM) = TENDENCY_LOC_CLD(JL, JK, JM) + (ZQXN(JM) - ZQX0(JL, JK, JM))*ZQTMST

        END DO

        !----------------------
        ! 6.2 Humidity budget
        !----------------------
        TENDENCY_LOC_q(JL, JK) = TENDENCY_LOC_Q(JL, JK) + (ZQXN(NCLDQV) - ZQX(JL, JK, NCLDQV))*ZQTMST

        !-------------------
        ! 6.3 cloud cover
        !-----------------------
        TENDENCY_LOC_a(JL, JK) = TENDENCY_LOC_A(JL, JK) + ZDA*ZQTMST

        !--------------------------------------------------
        ! Copy precipitation fraction into output variable
        !-------------------------------------------------
        PCOVPTOT(JL, JK) = ZCOVPTOT

      END DO
      ! on vertical level JK
      !----------------------------------------------------------------------
      !                       END OF VERTICAL LOOP
      !----------------------------------------------------------------------

      !######################################################################
      !              8  *** FLUX/DIAGNOSTICS COMPUTATIONS ***
      !######################################################################

      !--------------------------------------------------------------------
      ! Copy general precip arrays back into PFP arrays for GRIB archiving
      ! Add rain and liquid fluxes, ice and snow fluxes
      !--------------------------------------------------------------------
      DO JK=1,KLEV + 1
        PFPLSL(JL, JK) = ZPFPLSX(JL, JK, NCLDQR) + ZPFPLSX(JL, JK, NCLDQL)
        PFPLSN(JL, JK) = ZPFPLSX(JL, JK, NCLDQS) + ZPFPLSX(JL, JK, NCLDQI)
      END DO

      !--------
      ! Fluxes:
      !--------
      PFSQLF(JL, 1) = 0.0_JPRB
      PFSQIF(JL, 1) = 0.0_JPRB
      PFSQRF(JL, 1) = 0.0_JPRB
      PFSQSF(JL, 1) = 0.0_JPRB
      PFCQLNG(JL, 1) = 0.0_JPRB
      PFCQNNG(JL, 1) = 0.0_JPRB
      PFCQRNG(JL, 1) = 0.0_JPRB        !rain
      PFCQSNG(JL, 1) = 0.0_JPRB        !snow
      ! fluxes due to turbulence
      PFSQLTUR(JL, 1) = 0.0_JPRB
      PFSQITUR(JL, 1) = 0.0_JPRB

      DO JK=1,KLEV

        ZGDPH_R = -ZRG_R*(PAPH(JL, JK + 1) - PAPH(JL, JK))*ZQTMST
        PFSQLF(JL, JK + 1) = PFSQLF(JL, JK)
        PFSQIF(JL, JK + 1) = PFSQIF(JL, JK)
        PFSQRF(JL, JK + 1) = PFSQLF(JL, JK)
        PFSQSF(JL, JK + 1) = PFSQIF(JL, JK)
        PFCQLNG(JL, JK + 1) = PFCQLNG(JL, JK)
        PFCQNNG(JL, JK + 1) = PFCQNNG(JL, JK)
        PFCQRNG(JL, JK + 1) = PFCQLNG(JL, JK)
        PFCQSNG(JL, JK + 1) = PFCQNNG(JL, JK)
        PFSQLTUR(JL, JK + 1) = PFSQLTUR(JL, JK)
        PFSQITUR(JL, JK + 1) = PFSQITUR(JL, JK)

        ZALFAW = ZFOEALFA(JL, JK)

        ! Liquid , LS scheme minus detrainment
        PFSQLF(JL, JK + 1) = PFSQLF(JL, JK + 1) + (ZQXN2D(JL, JK, NCLDQL) - ZQX0(JL, JK, NCLDQL) + PVFL(JL, JK)*PTSPHY -  &
        & ZALFAW*PLUDE(JL, JK))*ZGDPH_R
        ! liquid, negative numbers
        PFCQLNG(JL, JK + 1) = PFCQLNG(JL, JK + 1) + ZLNEG(JL, JK, NCLDQL)*ZGDPH_R

        ! liquid, vertical diffusion
        PFSQLTUR(JL, JK + 1) = PFSQLTUR(JL, JK + 1) + PVFL(JL, JK)*PTSPHY*ZGDPH_R

        ! Rain, LS scheme
        PFSQRF(JL, JK + 1) = PFSQRF(JL, JK + 1) + (ZQXN2D(JL, JK, NCLDQR) - ZQX0(JL, JK, NCLDQR))*ZGDPH_R
        ! rain, negative numbers
        PFCQRNG(JL, JK + 1) = PFCQRNG(JL, JK + 1) + ZLNEG(JL, JK, NCLDQR)*ZGDPH_R

        ! Ice , LS scheme minus detrainment
        PFSQIF(JL, JK + 1) = PFSQIF(JL, JK + 1) + (ZQXN2D(JL, JK, NCLDQI) - ZQX0(JL, JK, NCLDQI) + PVFI(JL, JK)*PTSPHY -  &
        & (1.0_JPRB - ZALFAW)*PLUDE(JL, JK))*ZGDPH_R
        ! ice, negative numbers
        PFCQNNG(JL, JK + 1) = PFCQNNG(JL, JK + 1) + ZLNEG(JL, JK, NCLDQI)*ZGDPH_R

        ! ice, vertical diffusion
        PFSQITUR(JL, JK + 1) = PFSQITUR(JL, JK + 1) + PVFI(JL, JK)*PTSPHY*ZGDPH_R

        ! snow, LS scheme
        PFSQSF(JL, JK + 1) = PFSQSF(JL, JK + 1) + (ZQXN2D(JL, JK, NCLDQS) - ZQX0(JL, JK, NCLDQS))*ZGDPH_R
        ! snow, negative numbers
        PFCQSNG(JL, JK + 1) = PFCQSNG(JL, JK + 1) + ZLNEG(JL, JK, NCLDQS)*ZGDPH_R
      END DO

      !-----------------------------------
      ! enthalpy flux due to precipitation
      !-----------------------------------
      DO JK=1,KLEV + 1
        PFHPSL(JL, JK) = -RLVTT*PFPLSL(JL, JK)
        PFHPSN(JL, JK) = -RLSTT*PFPLSN(JL, JK)
      END DO

      !===============================================================================
      !IF (LHOOK) CALL DR_HOOK('CLOUDSC',1,ZHOOK_HANDLE)
    !END DO
  END SUBROUTINE CLOUDSC_SCC
END MODULE CLOUDSC_GPU_OMP_SCC_MOD
