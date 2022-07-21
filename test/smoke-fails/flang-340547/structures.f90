!
!  Copyright 2019-2020 SALMON developers
!
!  Licensed under the Apache License, Version 2.0 (the "License");
!  you may not use this file except in compliance with the License.
!  You may obtain a copy of the License at
!
!      http://www.apache.org/licenses/LICENSE-2.0
!
!  Unless required by applicable law or agreed to in writing, software
!  distributed under the License is distributed on an "AS IS" BASIS,
!  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
!  See the License for the specific language governing permissions and
!  limitations under the License.
!
!--------10--------20--------30--------40--------50--------60--------70--------80--------90--------100-------110-------120-------130


module structures

  implicit none

! scalar field
  type s_scalar
    real(8),allocatable :: f(:,:,:) ! f(x,y,z)
  end type s_scalar

! vector field
  type s_vector
    real(8),allocatable :: v(:,:,:,:) ! v(1:3,x,y,z)
  end type s_vector

  type s_dft_system
    logical :: if_real_orbital
    integer :: ngrid,nspin,no,nk,nion ! # of r-grid points, spin indices, orbitals, k points, and ions
    real(8) :: hvol,hgs(3),primitive_a(3,3),det_a,primitive_b(3,3)
    real(8) :: rmatrix_a(3,3),rmatrix_b(3,3)
    real(8) :: mu
    real(8),allocatable :: vec_k(:,:)    ! (1:3,1:nk), k-vector
    real(8),allocatable :: wtk(:)        ! (1:nk), weight of k points
    real(8),allocatable :: rocc(:,:,:)   ! (1:no,1:nk,1:nspin), occupation rate
  ! atomic ...
    real(8),allocatable :: Mass(:)       ! (1:nelem), Atomic weight
    real(8),allocatable :: Rion(:,:)     ! (1:3,1:nion), atom position
    real(8),allocatable :: Velocity(:,:) ! (1:3,1:nion), atomic velocity
    real(8),allocatable :: Force(:,:)    ! (1:3,1:nion), force on atom
  ! external field
    real(8) :: vec_Ac(3) ! A/c (spatially averaged), A: vector potential, c: speed of light
    type(s_vector) :: Ac_micro ! A/c (microscopic)      ! for single-scale Maxwell-TDDFT
    type(s_scalar) :: div_Ac   ! divergence of Ac_micro ! for single-scale Maxwell-TDDFT
    real(8) :: vec_Ac_ext(3) ! external vector potential for output
    real(8) :: vec_E(3)      ! total electric field for output
    real(8) :: vec_E_ext(3)  ! external electric potential for output
  end type s_dft_system

  type s_dft_energy
    real(8),allocatable :: esp(:,:,:) ! (1:no,1:nk,1:nspin), single-particle energy
    real(8) :: E_tot,E_kin,E_h,E_xc,E_ion_ion,E_ion_loc,E_ion_nloc
    real(8) :: E_U
    real(8) :: E_tot0 ! total energy @ t=0
  end type s_dft_energy

  type s_ewald_ion_ion
    integer :: nmax_pair_bk
    integer,allocatable :: bk(:,:,:) ! left  :1-3=ix,iy,iz, 4=pair atom
                                     ! middle:ion-ion pair index
                                     ! right :atom id
    integer,allocatable :: g(:,:)
    integer,allocatable :: npair_bk(:)
    integer :: ng_bk, ng_r, ng_s, ng_e
    character(1) :: yn_bookkeep
  end type s_ewald_ion_ion

  type s_rgrid
    integer              :: ndir,Nd                 ! ndir=3 --> dir=xx,yy,zz, ndir=6 --> dir=xx,yy,zz,yz,zx,xy
    integer,dimension(3) :: is,ie,num &             ! num=ie-is+1
                           ,is_overlap,ie_overlap & ! is_overlap=is-Nd, ie_overlap=ie+Nd
                           ,is_array,ie_array       ! allocate( array(is_array(1):ie_array(1), ...) )
    integer ,allocatable :: idx(:),idy(:),idz(:)    ! idx(is_overlap(1):ie_overlap(1))=is_array(1)~ie_array(1), ...
    integer ,allocatable :: is_all(:,:),ie_all(:,:) ! (1:3,0:nproc-1), is & ie for all MPI processes
    real(8) ,allocatable :: coordinate(:,:)         ! (minval(is_overlap):maxval(ie_overlap),1:3), coordinate of grids
  end type s_rgrid

! for persistent communication
  type s_pcomm_cache
    real(8), allocatable :: dbuf(:, :, :, :)
    complex(8), allocatable :: zbuf(:, :, :, :)
#ifdef FORTRAN_COMPILER_HAS_2MB_ALIGNED_ALLOCATION
!dir$ attributes align : 2097152 :: dbuf, zbuf
#endif
  end type s_pcomm_cache

! for update_overlap
  type s_sendrecv_grid
    ! Number of orbitals (4-th dimension of grid)
    integer :: nb
    ! Communicator
    integer :: icomm
    ! Neightboring MPI id (1:upside,2:downside, 1:x,2:y,3:z):
    integer :: neig(1:2, 1:3)
    ! Communication requests (1:send,2:recv, 1:upside,2:downside, 1:x,2:y,3:z):
    integer :: ireq_real8(1:2, 1:2, 1:3)
    integer :: ireq_complex8(1:2, 1:2, 1:3)
    ! PComm cache (1:src/2:dst, 1:upside,2:downside, 1:x,2:y,3:z)
    type(s_pcomm_cache) :: cache(1:2, 1:2, 1:3)
    ! Range (axis=1...3, 1:src/2:dst, dir=1:upside,2:downside, dim=1:x,2:y,3:z)
    integer :: is_block(1:3, 1:2, 1:2, 1:3)
    integer :: ie_block(1:3, 1:2, 1:2, 1:3)
    ! Initialization flags
    logical :: if_pcomm_real8_initialized
    logical :: if_pcomm_complex8_initialized
  end type s_sendrecv_grid

  type s_parallel_info
  ! division of MPI processes (for orbital wavefunction)
    integer :: npk        ! k-points
    integer :: nporbital  ! orbital index
    integer :: nprgrid(3) ! r-space (x,y,z)
  ! parallelization of orbital wavefunction
    integer :: iaddress(5) ! address of MPI under orbital wavefunction (ix,iy,iz,io,ik)
    integer,allocatable :: imap(:,:,:,:,:) ! address map
    integer :: iaddress_isolated_ffte(7) ! address of MPI for isolated_ffte (ix,iy,iz,io1,io2,io3,ik)
    integer,allocatable :: imap_isolated_ffte(:,:,:,:,:,:,:) ! address map for isolated_ffte 
    logical :: if_divide_rspace
    logical :: if_divide_orbit
    integer :: icomm_r,   id_r,   isize_r   ! communicator, process ID, & # of processes for r-space
    integer :: icomm_k,   id_k,   isize_k   ! communicator, process ID, & # of processes for k-space
    integer :: icomm_o,   id_o,   isize_o   ! communicator, process ID, & # of processes for orbital
    integer :: icomm_ro,  id_ro,  isize_ro  ! communicator, process ID, & # of processes for r-space & orbital
    integer :: icomm_ko,  id_ko,  isize_ko  ! communicator, process ID, & # of processes for k-space & orbital
    integer :: icomm_rko, id_rko, isize_rko ! communicator, process ID, & # of processes for r-space, k-space & orbital
    integer :: im_s,im_e,numm ! im=im_s,...,im_e, numm=im_e-im_s+1
    integer :: ik_s,ik_e,numk ! ik=ik_s,...,ik_e, numk=ik_e-ik_s+1
    integer :: io_s,io_e,numo ! io=io_s,...,io_e, numo=io_e-io_s+1
                              ! For calc_mode='RT' and temperature<0, these values are calculated from nelec.
                              ! In other cases, these are calculated from nstate.
  ! sub-communicators of icomm_r (r-space)
    integer :: icomm_x,id_x,isize_x ! x-axis
    integer :: icomm_y,id_y,isize_y ! y-axis
    integer :: icomm_z,id_z,isize_z ! z-axis
    integer :: icomm_xy,id_xy,isize_xy ! for singlescale FDTD (and for FFTW)
  ! for atom index #ia
    integer :: ia_s,ia_e ! ia=ia_s,...,ia_e
    integer :: nion_mg
    integer,allocatable :: ia_mg(:)
  ! for orbital index #io
    integer,allocatable :: irank_io(:) ! MPI rank of the orbital index #io
    integer,allocatable :: io_s_all(:) ! io_s for all orbital ranks
    integer,allocatable :: io_e_all(:) ! io_e for all orbital ranks
    integer,allocatable :: numo_all(:) ! numo for all orbital ranks
    integer :: numo_max ! max value of numo_all
#ifdef USE_SCALAPACK
    logical :: flag_blacs_gridinit
    integer :: icomm_sl ! for summation
    integer :: iam,nprocs
    integer,allocatable :: gridmap(:,:)
    integer :: nprow,npcol,myrow,mycol
    integer :: nrow_local,ncol_local,lda
    integer :: desca(9), descz(9)
    integer :: len_work  ! for PDSYEVD, PZHEEVD
    integer :: len_rwork ! for PZHEEVD
    integer,allocatable :: ndiv(:), i_tbl(:,:), j_tbl(:,:), iloc_tbl(:,:), jloc_tbl(:,:)
#endif
#ifdef USE_EIGENEXA
    logical :: flag_eigenexa_init
#endif
    integer :: icomm_x_isolated_ffte,id_x_isolated_ffte,isize_x_isolated_ffte ! x-axis for isolated_ffte
    integer :: icomm_y_isolated_ffte,id_y_isolated_ffte,isize_y_isolated_ffte ! y-axis for isolated_ffte
    integer :: icomm_z_isolated_ffte,id_z_isolated_ffte,isize_z_isolated_ffte ! z-axis for isolated_ffte
    integer :: icomm_o_isolated_ffte,id_o_isolated_ffte,isize_o_isolated_ffte ! o-axis for isolated_ffte
#ifdef USE_FFTW
    integer :: iaddress_isolated_fftw(6) ! address of MPI for isolated_ffte (ix,iy,iz,io3,io4,ik)
    integer,allocatable :: imap_isolated_fftw(:,:,:,:,:,:) ! address map for isolated_fftw 
    integer :: icomm_z_isolated_fftw,id_z_isolated_fftw,isize_z_isolated_fftw ! z-axis for isolated_fftw
    integer :: icomm_o_isolated_fftw,id_o_isolated_fftw,isize_o_isolated_fftw ! o-axis for isolated_fftw
#endif
  end type s_parallel_info

  type s_orbital
  ! ispin=1~nspin, io=io_s~io_e, ik=ik_s~ik_e, im=im_s~im_e (cf. s_parallel_info)
    real(8)   ,allocatable :: rwf(:,:,:,:,:,:,:) ! (ix,iy,iz,ispin,io,ik,im)
    complex(8),allocatable :: zwf(:,:,:,:,:,:,:) ! (ix,iy,iz,ispin,io,ik,im)
    logical :: update_zwf_overlap   !flag of update_overlap_complex8 for zwf
  end type s_orbital

  type s_stencil
    logical :: if_orthogonal
    real(8) :: coef_lap0,coef_lap(4,3),coef_nab(4,3) ! (4,3) --> (Nd,3) (future work)
    real(8) :: coef_f(6) ! for non-orthogonal lattice
  end type s_stencil

! pseudopotential
  type s_pp_info
    real(8) :: zion
    integer :: lmax,lmax0
    integer :: nrmax,nrmax0
    logical :: flag_nlcc
    character(2),allocatable :: atom_symbol(:)
    real(8),allocatable :: rmass(:)
    integer,allocatable :: mr(:)
    integer,allocatable :: lref(:)
    integer,allocatable :: nrps(:)
    integer,allocatable :: mlps(:)
    integer,allocatable :: nproj(:,:)
    integer,allocatable :: num_orb(:)
    integer,allocatable :: zps(:)
    integer,allocatable :: nrloc(:)
    real(8),allocatable :: rloc(:)
    real(8),allocatable :: rps(:)
    real(8),allocatable :: anorm(:,:)
    integer,allocatable :: inorm(:,:)
    real(8),allocatable :: anorm_so(:,:) ! '*_so' means what is used in
    integer,allocatable :: inorm_so(:,:) !   spin-orbit calculation
    real(8),allocatable :: rad(:,:)
    real(8),allocatable :: radnl(:,:)
    real(8),allocatable :: vloctbl(:,:)
    real(8),allocatable :: dvloctbl(:,:)
    real(8),allocatable :: udvtbl(:,:,:)
    real(8),allocatable :: dudvtbl(:,:,:)
    real(8),allocatable :: rho_pp_tbl(:,:)
    real(8),allocatable :: rho_nlcc_tbl(:,:)
    real(8),allocatable :: tau_nlcc_tbl(:,:)
    real(8),allocatable :: upp_f(:,:,:)
    real(8),allocatable :: vpp_f(:,:,:)
    real(8),allocatable :: vpp_f_so(:,:,:)
    real(8),allocatable :: upp(:,:)
    real(8),allocatable :: dupp(:,:)
    real(8),allocatable :: vpp(:,:)
    real(8),allocatable :: dvpp(:,:)
    real(8),allocatable :: vpp_so(:,:)
    real(8),allocatable :: dvpp_so(:,:)
    real(8),allocatable :: udvtbl_so(:,:,:)
    real(8),allocatable :: dudvtbl_so(:,:,:)
    real(8),allocatable :: rps_ao(:)
    integer,allocatable :: nrps_ao(:)
    real(8),allocatable :: upptbl_ao(:,:,:)
    real(8),allocatable :: dupptbl_ao(:,:,:)
  end type s_pp_info

! pseudopotential on r-space grid
  type s_pp_grid
    integer :: nps
    integer,allocatable :: mps(:)
    integer,allocatable :: jxyz(:,:,:)
    real(8),allocatable :: rxyz(:,:,:)
    real(8),allocatable :: uv(:,:)
    real(8),allocatable :: duv(:,:,:)
    integer :: nlma
    integer,allocatable :: lma_tbl(:,:)
    integer,allocatable :: ia_tbl(:)
    real(8),allocatable :: rinv_uvu(:)
    complex(8),allocatable :: zekr_uv(:,:,:) ! (j,ilma,ik), j=1~Mps(ia), ilma=1~Nlma, zekr_uV = exp(-i(k+A/c)r)*uv
    !
    complex(8),allocatable :: zrhoG_ion(:,:,:),zVG_ion(:,:,:,:) ! rho_ion(G),V_ion(G): local part of pseudopotential in G-space
    real(8),allocatable :: Vpsl_ion(:,:,:,:) ! local part of pseudopotential in r-space (isolated system)
    !
    integer,allocatable :: ia_tbl_so(:)
    real(8),allocatable :: rinv_uvu_so(:)
    complex(8),allocatable :: uv_so(:,:,:,:)
    complex(8),allocatable :: duv_so(:,:,:,:,:)
    complex(8),allocatable :: zekr_uv_so(:,:,:,:,:)
    !
    real(8),allocatable :: Rion_old(:,:) ! old position
    integer,allocatable :: jxyz_old(:,:,:)
    integer,allocatable :: jxx_old(:,:)
    integer,allocatable :: jyy_old(:,:)
    integer,allocatable :: jzz_old(:,:)
    integer,allocatable :: mps_old(:)
    real(8),allocatable :: rxyz_old(:,:,:)
    integer,allocatable :: jxyz_min(:,:)
    integer,allocatable :: jxyz_max(:,:)
    logical,allocatable :: jxyz_changed(:)
    !
    integer,allocatable :: proj_pairs_ao(:,:)
    integer,allocatable :: proj_pairs_info_ao(:,:)
    integer,allocatable :: ia_tbl_ao(:)
    real(8),allocatable :: phi_ao(:,:)
    real(8),allocatable :: dphi_ao(:,:,:)
    complex(8),allocatable :: zekr_phi_ao(:,:,:)
    integer :: nps_ao
    integer,allocatable :: mps_ao(:)
    integer,allocatable :: jxyz_ao(:,:,:)
    integer,allocatable :: jxx_ao(:,:)
    integer,allocatable :: jyy_ao(:,:)
    integer,allocatable :: jzz_ao(:,:)
    real(8),allocatable :: rxyz_ao(:,:,:)
    ! for localized communication when calculating non-local pseudo-pt.
    integer,allocatable :: irange_atom(:,:)    ! uVpsi range for atom: n = (1,ia), m = (2,ia)
    logical,allocatable :: ireferred_atom(:)   ! uVpsi(n:m) is referred in this process
    logical,allocatable :: ireferred_atom_comm_r(:,:)
    integer,allocatable :: icomm_atom(:)       ! communicator for uVpsi(n:m)
    ! for reducing pseudo-potential parallelization costs.
    integer             :: ilocal_nlma         ! number of own nlma
    integer,allocatable :: ilocal_nlma2ilma(:) ! ilocal_nlma to global nlma
    integer,allocatable :: ilocal_nlma2ia(:)   ! ilocal_nlma to atom number (ia_tbl)
    ! for optimizing OpenACC
    complex(8),allocatable :: uVpsibox(:,:,:,:,:)
    integer                :: max_vi
    integer,allocatable    :: v2nlma(:)
    integer,allocatable    :: k2ilma(:,:)
    integer,allocatable    :: k2j(:,:)
    integer,allocatable    :: v2j(:,:)
    !
    real(8),allocatable :: save_udVtbl_a(:,:,:)
    real(8),allocatable :: save_udVtbl_b(:,:,:)
    real(8),allocatable :: save_udVtbl_c(:,:,:)
    real(8),allocatable :: save_udVtbl_d(:,:,:)
  end type s_pp_grid

  type s_pp_nlcc
    real(8), allocatable :: rho_nlcc(:,:,:)
    real(8), allocatable :: tau_nlcc(:,:,:)
  end type s_pp_nlcc

! exchange-correlation functional
  type s_xc_functional
    integer :: xctype(3)
    integer :: ispin
    real(8) :: cval
    logical :: use_gradient
    logical :: use_laplacian
    logical :: use_kinetic_energy
    logical :: use_current
#ifdef USE_LIBXC
#if XC_MAJOR_VERSION <= 4 
    type(xc_f90_pointer_t) :: func(3)
    type(xc_f90_pointer_t) :: info(3)
#else
    TYPE(xc_f90_func_t) :: func(3)
    TYPE(xc_f90_func_info_t) :: info(3)
#endif
#endif
  end type

  type s_reciprocal_grid
    logical,allocatable :: if_Gzero(:,:,:)
    real(8),allocatable :: vec_G(:,:,:,:)   ! G vector (reciprocal lattice vector)
    real(8),allocatable :: coef(:,:,:)      ! 4*pi/|G|^2 (coefficient of the Poisson equation)
    real(8),allocatable :: exp_ewald(:,:,:) ! exp(-|G|^2/(4*a_Ewald))
    complex(8),allocatable :: egx(:,:),egxc(:,:),egy(:,:),egyc(:,:),egz(:,:),egzc(:,:)
    complex(8),allocatable :: coef_nabla(:,:,:,:),coef_gxgy0(:,:,:),cos_cGdt(:,:,:),sin_cGdt(:,:,:) ! for single-scale Maxwell-TDDFT
  end type s_reciprocal_grid

! Poisson equation
  type s_poisson
  ! for poisson_isolated_cg (conjugate-gradient method)
    integer :: iterVh                              ! iteration number for poisson_isolated_cg
    integer :: npole_partial                       ! number of multipoles calculated in each node
    integer :: npole_total                         ! total number of multipoles
    integer,allocatable :: ipole_tbl(:)            ! table for multipoles
    integer,allocatable :: ig_num(:)               ! number of grids for domains to which each multipole belongs
    integer,allocatable :: ig(:,:,:)               ! grid table for domains to which each multipole belongs
    integer,allocatable :: ig_bound(:,:,:)         ! grid table for boundaries
    real(8),allocatable :: wkbound(:), wkbound2(:) ! values on boundary represented in one-dimentional grid
    integer :: n_multipole_xyz(3)                  ! number of multipoles
  ! for Fourier transform
    complex(8),allocatable :: zrhoG_ele(:,:,:)     ! rho_ele(G): Fourier transform of the electron density
  ! for discrete Fourier transform (general)
    complex(8),allocatable :: ff1x(:,:,:),ff1y(:,:,:),ff1z(:,:,:),ff2x(:,:,:),ff2y(:,:,:),ff2z(:,:,:)
    complex(8),allocatable :: ff1(:,:,:),ff2(:,:,:) ! for isolated_ffte
    complex(8),allocatable :: ff3x(:,:,:),ff3y(:,:,:),ff3z(:,:,:),ff4x(:,:,:),ff4y(:,:,:),ff4z(:,:,:) ! for isolated_ffte
  ! for FFTE
    complex(8),allocatable :: a_ffte(:,:,:),b_ffte(:,:,:)
#ifdef USE_FFTW
  ! for FFTW
    complex(8),allocatable :: fftw1(:,:,:),fftw2(:,:,:)
#endif
  end type s_poisson

  type s_fdtd_system
    type(s_rgrid)         :: lg, mg       ! Structure for send and receive in fdtd
    type(s_sendrecv_grid) :: srg_ng       ! Structure for send and receive in fdtd
    real(8) :: rlsize(3)                  ! Size of Cell
    real(8) :: hgs(3)                     ! Grid Spacing
    real(8) :: origin(3)                  ! Coordinate of Origin Point (TBA)
    character(8)  :: a_bc(3,2)            ! Boundary Condition for 1:x, 2:y, 3:z and 1:bottom and 2:top
    integer, allocatable :: imedia(:,:,:) ! Material information
  end type s_fdtd_system

  type s_opt
     real(8),allocatable :: a_dRion(:), dFion(:)
     real(8),allocatable :: Hess_mat(:,:), Hess_mat_last(:,:)
  end type s_opt

  type s_md
    real(8) :: Uene,  E_tot, Tene, E_work, E_nh, Htot
    real(8) :: Uene0, E_tot0
    real(8) :: Enh_gkTlns, xi_nh, Qnh, gkT
    real(8) :: Temperature
    real(8),allocatable :: Rion_last(:,:), Force_last(:,:)
  end type s_md

! output files
  type s_ofile
    integer :: fh_eigen
    integer :: fh_rt
    integer :: fh_rt_energy
    integer :: fh_response
    integer :: fh_pulse
    integer :: fh_dft_md
    integer :: fh_ovlp,fh_nex
    integer :: fh_mag,fh_gs_mag
    character(100) :: file_eigen_data
    character(256) :: file_rt_data
    character(256) :: file_rt_energy_data
    character(256) :: file_response_data
    character(256) :: file_pulse_data
    character(256) :: file_dft_md
    character(256) :: file_ovlp,file_nex
    character(256) :: file_mag, file_gs_mag
    !
    character(256) :: dir_out_restart, dir_out_checkpoint
  end type s_ofile

! +-----------------------------------+
! | for DFT ground-state calculations |
! +-----------------------------------+

  type s_cg
    type(s_orbital) :: xk,hxk,gk,pk,pko,hwf
  end type s_cg

  type s_mixing
    logical :: flag_mix_zero
    integer :: num_rho_stock
    type(s_scalar),allocatable :: rho_in(:), rho_out(:), rho_s_in(:,:), rho_s_out(:,:)
    real(8) :: mixrate, alpha_mb, beta_p
    real(8) :: convergence_value_prev
  end type s_mixing

  type s_band_dft
    integer :: num_band_kpt, nref_band
    real(8),allocatable :: band_kpt(:,:)
    logical,allocatable :: check_conv_esp(:,:,:)
  end type s_band_dft

! DFT_k_expand
!(k-expand; just for gbp, temporal)
  type s_k_expand
     integer :: nk, nkx,nky,nkz
     integer,allocatable :: isupercell(:,:)
     real(8),allocatable :: k_vec(:,:)
     integer :: nk_new, no_new, nmax
     integer :: natom, num_rgrid(3), nelec, nstate
     real(8) :: al(3)
     integer,allocatable :: myrank(:), iaddress(:,:), iaddress_new(:,:)
  end type s_k_expand

! +----------------------------------+
! | for TDDFT real-time calculations |
! +----------------------------------+

  type s_rt
    real(8) :: Dp0_e(3)     !rename later
    real(8), allocatable :: dDp_e(:,:), Dp_e(:,:), Dp_i(:,:), Qp_e(:,:,:)    !rename later
    real(8), allocatable :: rIe(:)    !rename later
    real(8), allocatable :: curr(:,:), E_ext(:,:), E_ind(:,:), E_tot(:,:)
    real(8), allocatable :: Ac_ext(:,:), Ac_ind(:,:), Ac_tot(:,:)
    complex(8), allocatable :: zc(:)
    type(s_scalar),allocatable :: vloc_t(:), vloc_new(:)
    type(s_scalar),allocatable :: vloc_old(:,:)  ! vloc_old(spin,iteration)
    type(s_scalar),allocatable :: rho0_s(:) ! =rho_s(1:nspin) @ t=0 (GS)
    type(s_scalar) :: vonf
    type(s_vector) :: j_e ! microscopic electron number current density
    ! for projection_option
    type(s_dft_system) :: system_gs
    type(s_parallel_info) :: info_gs
    type(s_scalar),allocatable :: vloc0(:)  ! =v_local(1:nspin) @ t=0 (GS)
    type(s_orbital) :: tpsi0,ttpsi0,htpsi0
    type(s_cg) :: cg
    real(8) :: E_old
  end type s_rt

! single-scale Maxwell-TDDFT method
  type s_singlescale
    logical :: flag_use
    integer :: fh_rt_micro,fh_excitation,fh_Ac_zt
    real(8) :: E_electron,Energy_joule,Energy_poynting(2),coef_nab(4,3),curr_ave(3)
    real(8),allocatable :: vec_Ac_old(:,:,:,:),vec_Ac_m(:,:,:,:,:) &
    & ,curr(:,:,:,:),vec_je_old(:,:,:,:),rho_old(:,:,:) &
    & ,current4pi(:,:,:,:),grad_Vh(:,:,:,:),grad_Vh_old(:,:,:,:) &
    & ,vec_Ac_boundary_bottom(:,:,:),vec_Ac_boundary_bottom_old(:,:,:) &
    & ,vec_Ac_boundary_top(:,:,:),vec_Ac_boundary_top_old(:,:,:) &
    & ,integral_poynting(:),Ac_zt(:,:),tmp_zt(:,:)
    real(8),allocatable :: box(:,:,:),box1(:,:,:),rot_Ac(:,:,:,:),poynting_vector(:,:,:,:) &
    & ,div_Ac(:,:,:),div_Ac_old(:,:,:) &
    & ,integral_poynting_tmp(:),integral_poynting_tmp2(:)
    type(s_sendrecv_grid) :: srg_eg ! specialized in FDTD timestep
    type(s_rgrid)         :: eg
  ! for method_singlescale=='1d','1d_fourier'
    real(8),dimension(3) :: Ac_zt_boundary_bottom,Ac_zt_boundary_top,Ac_zt_boundary_bottom_old,Ac_zt_boundary_top_old
    real(8),allocatable :: curr4pi_zt(:,:),Ac_zt_m(:,:,:)
    real(8),allocatable :: Ac_fourier(:,:,:,:)
    complex(8),allocatable :: a_ffte(:,:,:,:),b_ffte(:,:,:,:),Vh_ffte_old(:,:,:),zf_old(:,:,:,:),zc_old(:,:,:,:),zs_old(:,:,:,:)
  end type s_singlescale


  
  type s_multiscale
    integer :: nmacro
    integer :: icomm_ms_world, isize_ms_world, id_ms_world ! Top level communicator
    integer :: icomm_macropoint, isize_macropoint, id_macropoint ! Macropoint communicator
    integer :: imacro_mygroup_s, imacro_mygroup_e
    integer :: id_mygroup_s, id_mygroup_e
    integer, allocatable :: ixyz_tbl(:, :)
    integer, allocatable :: imacro_tbl(:, :, :)
    character(256) :: base_directory
    character(256) :: base_directory_RT_Ac
    real(8), allocatable :: curr(:, :)
    real(8), allocatable :: vec_Ac(:, :)
    real(8), allocatable :: vec_Ac_old(:, :)
    character(256) :: directory_read_data
  end type s_multiscale

!===================================================================================================================================

contains

  subroutine allocate_scalar(rg,field)
    implicit none
    type(s_rgrid),intent(in) :: rg
    type(s_scalar)           :: field
    integer :: ix,iy,iz
    allocate(field%f(rg%is(1):rg%ie(1),rg%is(2):rg%ie(2),rg%is(3):rg%ie(3)))
!!$omp parallel do collapse(2) private(iz,iy,ix)
    do iz=rg%is(3),rg%ie(3)
    do iy=rg%is(2),rg%ie(2)
    do ix=rg%is(1),rg%ie(1)
      field%f(ix,iy,iz) = 0d0
    end do
    end do
    end do
  end subroutine allocate_scalar

  subroutine allocate_scalar_with_shadow(rg,nshadow,field)
    implicit none
    type(s_rgrid),intent(in) :: rg
    integer,intent(in)       :: nshadow
    type(s_scalar)           :: field
    integer :: ix,iy,iz
    allocate(field%f(rg%is(1)-nshadow:rg%ie(1)+nshadow &
                    ,rg%is(2)-nshadow:rg%ie(2)+nshadow &
                    ,rg%is(3)-nshadow:rg%ie(3)+nshadow))
!!$omp parallel do collapse(2) private(iz,iy,ix)
    do iz=rg%is(3)-nshadow,rg%ie(3)+nshadow
    do iy=rg%is(2)-nshadow,rg%ie(2)+nshadow
    do ix=rg%is(1)-nshadow,rg%ie(1)+nshadow
      field%f(ix,iy,iz) = 0d0
    end do
    end do
    end do
  end subroutine allocate_scalar_with_shadow

  subroutine allocate_vector(rg,field)
    implicit none
    type(s_rgrid),intent(in) :: rg
    type(s_vector)           :: field
    integer :: ix,iy,iz
    allocate(field%v(3,rg%is(1):rg%ie(1),rg%is(2):rg%ie(2),rg%is(3):rg%ie(3)))
!!$omp parallel do collapse(2) private(iz,iy,ix)
    do iz=rg%is(3),rg%ie(3)
    do iy=rg%is(2),rg%ie(2)
    do ix=rg%is(1),rg%ie(1)
      field%v(:,ix,iy,iz) = 0d0
    end do
    end do
    end do
  end subroutine allocate_vector

  subroutine allocate_vector_with_ovlp(rg,field)
    implicit none
    type(s_rgrid),intent(in) :: rg
    type(s_vector)           :: field
    integer :: ix,iy,iz
    allocate(field%v(3,rg%is_array(1):rg%ie_array(1),rg%is_array(2):rg%ie_array(2),rg%is_array(3):rg%ie_array(3)))
!!$omp parallel do collapse(2) private(iz,iy,ix)
    do iz=rg%is_array(3),rg%ie_array(3)
    do iy=rg%is_array(2),rg%ie_array(2)
    do ix=rg%is_array(1),rg%ie_array(1)
      field%v(:,ix,iy,iz) = 0d0
    end do
    end do
    end do
  end subroutine allocate_vector_with_ovlp

  subroutine allocate_orbital_real(nspin,mg,info,psi)
    implicit none
    integer                 ,intent(in) :: nspin
    type(s_rgrid)           ,intent(in) :: mg
    type(s_parallel_info)   ,intent(in) :: info
    type(s_orbital)                     :: psi
    integer :: im,ik,io,is,iz,iy,ix
    allocate(psi%rwf(mg%is_array(1):mg%ie_array(1),  &
                     mg%is_array(2):mg%ie_array(2),  &
                     mg%is_array(3):mg%ie_array(3),  &
                     nspin,info%io_s:info%io_e,info%ik_s:info%ik_e,info%im_s:info%im_e))
#ifdef USE_OPENACC
!$acc parallel loop collapse(6) private(im,ik,io,is,iz,iy,ix)
#else
!!$omp parallel do collapse(6) private(im,ik,io,is,iz,iy,ix)
#endif
    do im=info%im_s,info%im_e
    do ik=info%ik_s,info%ik_e
    do io=info%io_s,info%io_e
    do is=1,nspin
    do iz=mg%is_array(3),mg%ie_array(3)
    do iy=mg%is_array(2),mg%ie_array(2)
    do ix=mg%is_array(1),mg%ie_array(1)
      psi%rwf(ix,iy,iz,is,io,ik,im) = 0d0
    end do
    end do
    end do
    end do
    end do
    end do
    end do
  end subroutine allocate_orbital_real

  subroutine allocate_orbital_complex(nspin,mg,info,psi)
    implicit none
    integer                 ,intent(in) :: nspin
    type(s_rgrid)           ,intent(in) :: mg
    type(s_parallel_info)   ,intent(in) :: info
    type(s_orbital)                     :: psi
    integer :: im,ik,io,is,iz,iy,ix
    allocate(psi%zwf(mg%is_array(1):mg%ie_array(1),  &
                     mg%is_array(2):mg%ie_array(2),  &
                     mg%is_array(3):mg%ie_array(3),  &
                     nspin,info%io_s:info%io_e,info%ik_s:info%ik_e,info%im_s:info%im_e))
#ifdef USE_OPENACC
!$acc parallel loop collapse(6) private(im,ik,io,is,iz,iy,ix)
#else
!!$omp parallel do collapse(6) private(im,ik,io,is,iz,iy,ix)
#endif
    do im=info%im_s,info%im_e
    do ik=info%ik_s,info%ik_e
    do io=info%io_s,info%io_e
    do is=1,nspin
    do iz=mg%is_array(3),mg%ie_array(3)
    do iy=mg%is_array(2),mg%ie_array(2)
    do ix=mg%is_array(1),mg%ie_array(1)
      psi%zwf(ix,iy,iz,is,io,ik,im) = 0d0
    end do
    end do
    end do
    end do
    end do
    end do
    end do
  end subroutine allocate_orbital_complex

!===================================================================================================================================

# define DEAL(x) if(allocated(x)) deallocate(x)

  subroutine deallocate_dft_system(system)
    type(s_dft_system) :: system
    DEAL(system%rocc)
    DEAL(system%wtk)
    DEAL(system%Rion)
    DEAL(system%Velocity)
    DEAL(system%Force)
  end subroutine deallocate_dft_system

  subroutine deallocate_dft_energy(energy)
    type(s_dft_energy) :: energy
    DEAL(energy%esp)
  end subroutine deallocate_dft_energy

  subroutine deallocate_rgrid(rg)
    type(s_rgrid) :: rg
    DEAL(rg%idx)
    DEAL(rg%idy)
    DEAL(rg%idz)
  end subroutine deallocate_rgrid

  subroutine deallocate_orbital(psi)
    type(s_orbital) :: psi
    DEAL(psi%rwf)
    DEAL(psi%zwf)
  end subroutine deallocate_orbital

  subroutine deallocate_pp_info(pp)
    type(s_pp_info) :: pp
    DEAL(pp%atom_symbol)
    DEAL(pp%rmass)
    DEAL(pp%mr)
    DEAL(pp%lref)
    DEAL(pp%nrps)
    DEAL(pp%mlps)
    DEAL(pp%zps)
    DEAL(pp%nrloc)
    DEAL(pp%rloc)
    DEAL(pp%rps)
    DEAL(pp%anorm)
    DEAL(pp%inorm)
    DEAL(pp%rad)
    DEAL(pp%radnl)
    DEAL(pp%vloctbl)
    DEAL(pp%dvloctbl)
    DEAL(pp%udvtbl)
    DEAL(pp%dudvtbl)
    DEAL(pp%rho_nlcc_tbl)
    DEAL(pp%tau_nlcc_tbl)
    DEAL(pp%upp_f)
    DEAL(pp%vpp_f)
    DEAL(pp%upp)
    DEAL(pp%dupp)
    DEAL(pp%vpp)
    DEAL(pp%dvpp)
  end subroutine deallocate_pp_info

  subroutine deallocate_pp_grid(ppg)
    type(s_pp_grid) :: ppg
    DEAL(ppg%mps)
    DEAL(ppg%jxyz)
    DEAL(ppg%uv)
    DEAL(ppg%duv)
    DEAL(ppg%lma_tbl)
    DEAL(ppg%ia_tbl)
    DEAL(ppg%rinv_uvu)
    DEAL(ppg%zekr_uV)
    DEAL(ppg%uv_so)
    DEAL(ppg%duv_so)
  end subroutine deallocate_pp_grid

  subroutine deallocate_scalar(x)
    type(s_scalar) :: x
    DEAL(x%f)
  end subroutine deallocate_scalar

  subroutine deallocate_vector(x)
    type(s_vector) :: x
    DEAL(x%v)
  end subroutine deallocate_vector

end module structures
