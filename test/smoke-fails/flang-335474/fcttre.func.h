! (C) Copyright 1988- ECMWF.
!
! This software is licensed under the terms of the Apache Licence Version 2.0
! which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
!
! In applying this licence, ECMWF does not waive the privileges and immunities
! granted to it by virtue of its status as an intergovernmental organisation
! nor does it submit to any jurisdiction.

!*
!     ------------------------------------------------------------------

!     This COMDECK includes the Thermodynamical functions for the cy39
!       ECMWF Physics package.
!       Consistent with YOMCST Basic physics constants, assuming the
!       partial pressure of water vapour is given by a first order
!       Taylor expansion of Qs(T) w.r.t. to Temperature, using constants
!       in YOETHF
!       Two sets of functions are available. In the first set only the
!       cases water or ice are distinguished by temperature.  This set 
!       consists of the functions FOEDELTA,FOEEW,FOEDE and FOELH.
!       The second set considers, besides the two cases water and ice 
!       also a mix of both for the temperature range RTICE < T < RTWAT.
!       This set contains FOEALFA,FOEEWM,FOEDEM,FOELDCPM and FOELHM.
!       FKOOP modifies the ice saturation mixing ratio for homogeneous 
!       nucleation. FOE_DEWM_DT provides an approximate first derivative
!       of FOEEWM.

!       Depending on the consideration of mixed phases either the first 
!       set (e.g. surface, post-processing) or the second set 
!       (e.g. clouds, condensation, convection) should be used.

!     ------------------------------------------------------------------
!     *****************************************************************

!                NO CONSIDERATION OF MIXED PHASES

!     *****************************************************************
REAL(KIND=JPRB) :: FOEDELTA
REAL(KIND=JPRB) :: PTARE
FOEDELTA (PTARE) = MAX (0.0_4,SIGN(1.0_4,PTARE-RTT))

!                  FOEDELTA = 1    water
!                  FOEDELTA = 0    ice

!     THERMODYNAMICAL FUNCTIONS .

!     Pressure of water vapour at saturation
!        INPUT : PTARE = TEMPERATURE
REAL(KIND=JPRB) :: FOEEW,FOEDE,FOEDESU,FOELH,FOELDCP
FOEEW ( PTARE ) = R2ES*EXP (&
  &(R3LES*FOEDELTA(PTARE)+R3IES*(1.0_4-FOEDELTA(PTARE)))*(PTARE-RTT)&
&/ (PTARE-(R4LES*FOEDELTA(PTARE)+R4IES*(1.0_4-FOEDELTA(PTARE)))))

FOEDE ( PTARE ) = &
  &(FOEDELTA(PTARE)*R5ALVCP+(1.0_JPRB-FOEDELTA(PTARE))*R5ALSCP)&
&/ (PTARE-(R4LES*FOEDELTA(PTARE)+R4IES*(1.0_JPRB-FOEDELTA(PTARE))))**2

FOEDESU ( PTARE ) = &
  &(FOEDELTA(PTARE)*R5LES+(1.0_JPRB-FOEDELTA(PTARE))*R5IES)&
&/ (PTARE-(R4LES*FOEDELTA(PTARE)+R4IES*(1.0_JPRB-FOEDELTA(PTARE))))**2

FOELH ( PTARE ) =&
         &FOEDELTA(PTARE)*RLVTT + (1.0_JPRB-FOEDELTA(PTARE))*RLSTT

FOELDCP ( PTARE ) = &
         &FOEDELTA(PTARE)*RALVDCP + (1.0_JPRB-FOEDELTA(PTARE))*RALSDCP

!     *****************************************************************

!           CONSIDERATION OF MIXED PHASES

!     *****************************************************************

!     FOEALFA is calculated to distinguish the three cases:

!                       FOEALFA=1            water phase
!                       FOEALFA=0            ice phase
!                       0 < FOEALFA < 1      mixed phase

!               INPUT : PTARE = TEMPERATURE
REAL(KIND=JPRB) :: FOEALFA
FOEALFA (PTARE) = MIN(1.0_JPRB,((MAX(RTICE,MIN(RTWAT,PTARE))-RTICE)&
 &*RTWAT_RTICE_R)**2) 


!     Pressure of water vapour at saturation
!        INPUT : PTARE = TEMPERATURE
REAL(KIND=JPRB) :: FOEEWM,FOEDEM,FOELDCPM,FOELHM,FOE_DEWM_DT
FOEEWM ( PTARE ) = R2ES *&
     &(FOEALFA(PTARE)*EXP(R3LES*(PTARE-RTT)/(PTARE-R4LES))+&
  &(1.0_JPRB-FOEALFA(PTARE))*EXP(R3IES*(PTARE-RTT)/(PTARE-R4IES)))

FOE_DEWM_DT( PTARE ) = R2ES * ( &
     & R3LES*FOEALFA(PTARE)*EXP(R3LES*(PTARE-RTT)/(PTARE-R4LES)) &
     &    *(RTT-R4LES)/(PTARE-R4LES)**2 + &
     & R3IES*(1.0-FOEALFA(PTARE))*EXP(R3IES*(PTARE-RTT)/(PTARE-R4IES)) &
     &    *(RTT-R4IES)/(PTARE-R4IES)**2)

FOEDEM ( PTARE ) = FOEALFA(PTARE)*R5ALVCP*(1.0_JPRB/(PTARE-R4LES)**2)+&
             &(1.0_JPRB-FOEALFA(PTARE))*R5ALSCP*(1.0_JPRB/(PTARE-R4IES)**2)

FOELDCPM ( PTARE ) = FOEALFA(PTARE)*RALVDCP+&
            &(1.0_JPRB-FOEALFA(PTARE))*RALSDCP

FOELHM ( PTARE ) =&
         &FOEALFA(PTARE)*RLVTT+(1.0_JPRB-FOEALFA(PTARE))*RLSTT


!     Temperature normalization for humidity background change of variable
!        INPUT : PTARE = TEMPERATURE
REAL(KIND=JPRB) :: FOETB
FOETB ( PTARE )=FOEALFA(PTARE)*R3LES*(RTT-R4LES)*(1.0_JPRB/(PTARE-R4LES)**2)+&
             &(1.0_JPRB-FOEALFA(PTARE))*R3IES*(RTT-R4IES)*(1.0_JPRB/(PTARE-R4IES)**2)

!     ------------------------------------------------------------------
!     *****************************************************************

!           CONSIDERATION OF DIFFERENT MIXED PHASE FOR CONV

!     *****************************************************************

!     FOEALFCU is calculated to distinguish the three cases:

!                       FOEALFCU=1            water phase
!                       FOEALFCU=0            ice phase
!                       0 < FOEALFCU < 1      mixed phase

!               INPUT : PTARE = TEMPERATURE
REAL(KIND=JPRB) :: FOEALFCU 
FOEALFCU (PTARE) = MIN(1.0_JPRB,((MAX(RTICECU,MIN(RTWAT,PTARE))&
&-RTICECU)*RTWAT_RTICECU_R)**2) 


!     Pressure of water vapour at saturation
!        INPUT : PTARE = TEMPERATURE
REAL(KIND=JPRB) :: FOEEWMCU,FOEDEMCU,FOELDCPMCU,FOELHMCU
FOEEWMCU ( PTARE ) = R2ES *&
     &(FOEALFCU(PTARE)*EXP(R3LES*(PTARE-RTT)/(PTARE-R4LES))+&
  &(1.0_JPRB-FOEALFCU(PTARE))*EXP(R3IES*(PTARE-RTT)/(PTARE-R4IES)))

FOEDEMCU ( PTARE )=FOEALFCU(PTARE)*R5ALVCP*(1.0_JPRB/(PTARE-R4LES)**2)+&
             &(1.0_JPRB-FOEALFCU(PTARE))*R5ALSCP*(1.0_JPRB/(PTARE-R4IES)**2)

FOELDCPMCU ( PTARE ) = FOEALFCU(PTARE)*RALVDCP+&
            &(1.0_JPRB-FOEALFCU(PTARE))*RALSDCP

FOELHMCU ( PTARE ) =&
         &FOEALFCU(PTARE)*RLVTT+(1.0_JPRB-FOEALFCU(PTARE))*RLSTT
!     ------------------------------------------------------------------

!     Pressure of water vapour at saturation
!     This one is for the WMO definition of saturation, i.e. always
!     with respect to water.
!     
!     Duplicate to FOEELIQ and FOEEICE for separate ice variable
!     FOEELIQ always respect to water 
!     FOEEICE always respect to ice 
!     (could use FOEEW and FOEEWMO, but naming convention unclear)

REAL(KIND=JPRB) :: FOEEWMO, FOEELIQ, FOEEICE 
FOEEWMO( PTARE ) = R2ES*EXP(R3LES*(PTARE-RTT)/(PTARE-R4LES))
FOEELIQ( PTARE ) = R2ES*EXP(R3LES*(PTARE-RTT)/(PTARE-R4LES))
FOEEICE( PTARE ) = R2ES*EXP(R3IES*(PTARE-RTT)/(PTARE-R4IES))

REAL(KIND=JPRB) :: FOEEWM_V,FOEEWMCU_V,FOELES_V,FOEIES_V
REAL(KIND=JPRB) :: EXP1,EXP2
      FOELES_V(PTARE)=R3LES*(PTARE-RTT)/(PTARE-R4LES)
      FOEIES_V(PTARE)=R3IES*(PTARE-RTT)/(PTARE-R4IES)
      FOEEWM_V( PTARE,EXP1,EXP2 )=R2ES*(FOEALFA(PTARE)*EXP1+ &
          & (1.0_JPRB-FOEALFA(PTARE))*EXP2)
      FOEEWMCU_V ( PTARE,EXP1,EXP2 ) = R2ES*(FOEALFCU(PTARE)*EXP1+&
          &(1.0_JPRB-FOEALFCU(PTARE))*EXP2)

