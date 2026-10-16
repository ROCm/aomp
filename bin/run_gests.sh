#!/bin/bash
#
#  run_gests.sh
#
#    ./run_gests.sh [buildonly|runonly]
#
#  GESTS_NP, GESTS_JOBS, OMP_NUM_THREADS can be set.
#  The run input is $REPO_DIR/files/input.
#

# --- Start standard header to set AOMP environment variables ----
realpath=$(realpath "$0")
thisdir=$(dirname "$realpath")
export AOMP_USE_CCACHE=0
AOMP=${AOMP:-$HOME/rocm/aomp/llvm}
AOMP_SUPP=${AOMP_SUPP:-$HOME/local}
ROCM=${ROCM:-/opt/rocm}
HIPFORT_INCLUDE=${HIPFORT_INCLUDE:-$HOME/rocm/aomp}
HIPFORT_LIB=${HIPFORT_LIB:-$HOME/rocm/aomp/lib}
AOMPHIP=${AOMPHIP:-$(realpath -m "$(realpath -m "$AOMP")/../..")}

# shellcheck disable=SC1091
. "$thisdir/aomp_common_vars"
# --- end standard header ----

REPO_DIR=$AOMP_REPOS_TEST/GESTS
RUN_DIR=$AOMP_REPOS_TEST/run/GESTS
EXECUTABLE=DNS_PEN_GPU_p4.x

if [ -d "$REPO_DIR" ] ; then
  echo "$REPO_DIR exists."
else
  echo "Please contact Raghu Maddhipatla to get a copy of GESTS.tar and place the tar file at $HOME"
  exit
fi

# rocFFT is not in $ROCM. Use it from the system ROCm if needed.
if [ -f "$ROCM/lib/librocfft.so" ] ; then
  ROCFFT_LIB=$ROCM/lib
else
  ROCFFT_LIB=${ROCFFT_LIB:-/opt/rocm/lib}
fi

export FFTW_DIR=$AOMP_SUPP/fftw
export OPENMPI_DIR=$AOMP_SUPP/openmpi

echo "AOMP            = $AOMP"
echo "ROCM            = $ROCM"
echo "AOMP_SUPP       = $AOMP_SUPP"
echo "HIPFORT_INCLUDE = $HIPFORT_INCLUDE"
echo "HIPFORT_LIB     = $HIPFORT_LIB"
echo "AOMPHIP         = $AOMPHIP"

setaompgpu

if [ ! -d "$HIPFORT_INCLUDE" ] ; then
  echo "hipfort include directory not found at $HIPFORT_INCLUDE"
  exit 1
fi
if [ ! -f "$HIPFORT_LIB/libhipfort-amdgcn.a" ] ; then
  echo "missing $HIPFORT_LIB/libhipfort-amdgcn.a"
  exit 1
fi
if [ ! -d "$FFTW_DIR" ] ; then
  echo "$FFTW_DIR does not exist."
  exit 1
fi
if [ ! -d "$OPENMPI_DIR" ] ; then
  echo "$OPENMPI_DIR does not exist."
  exit 1
fi
if [ ! -f "$AOMP/bin/flang-new" ] ; then
  echo "flang-new not found in $AOMP/bin"
  exit 1
fi
if [ ! -f "$REPO_DIR/files/input" ] ; then
  echo "missing $REPO_DIR/files/input"
  exit 1
fi
if [ ! -f "$ROCFFT_LIB/librocfft.so" ] ; then
  echo "librocfft.so not found in $ROCFFT_LIB"
  exit 1
fi

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$AOMP/lib:$ROCM/lib:$ROCFFT_LIB:$OPENMPI_DIR/lib:$FFTW_DIR/lib
export PYTHONPATH=${PYTHONPATH:-$ROCM/share/amd_smi}
export HIP_CLANG_PATH=$AOMP/bin

GESTS_NP=${GESTS_NP:-1}
GESTS_JOBS=${GESTS_JOBS:-1}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}

if [ "$1" != "runonly" ] ; then
  echo "=================  STARTING MAKE in $REPO_DIR ========"
  cd "$REPO_DIR" || exit 1
  make clean
  make srcmake

  cppflags="-P -DOPENMP -DFFTW -DREVERSE_DIMS -DUSE_EVEN -DTIMERS -DDETAIL_MPI -DUSE_AMD -DUSE_HIPHOSTMALLOC -DPHYMAX -DMODEL_SPECTRUM"
  fflags="-cpp -I$HIPFORT_INCLUDE -I. -Ibin -Jbin -O1 --offload-arch=$AOMP_GPU -fopenmp"
  fft_inc="-I$FFTW_DIR/include"
  other_inc="-I$ROCM/include -I$HIPFORT_INCLUDE -I$OPENMPI_DIR/include -I$OPENMPI_DIR/lib"
  fft_lib="-L$FFTW_DIR/lib -lfftw3 -lfftw3f"
  other_libs="-L$OPENMPI_DIR/lib -lmpi -lmpi_usempif08 -lmpi_usempi_ignore_tkr -lmpi_mpifh -L$ROCM/lib -L$ROCFFT_LIB -L$HIPFORT_LIB -lrocfft -lamdhip64 -lhipfort-amdgcn -lhsa-runtime64 -lstdc++ -lm"

  make -j "$GESTS_JOBS" MACHINE=local \
      CPPFLAGS="$cppflags" \
      FF="$AOMP/bin/flang-new" \
      CPP=/lib/cpp \
      FFLAGS="$fflags" \
      FFLAGS90="$fflags" \
      FFT_INC="$fft_inc" \
      OTHER_INC="$other_inc" \
      FFT_LIB="$fft_lib" \
      OTHER_LIBS="$other_libs"
  rc=$?
  if [ $rc != 0 ] ; then
    echo "ERROR in Make"
    exit $rc
  fi
  if [ ! -f "$REPO_DIR/$EXECUTABLE" ] ; then
    echo "ERROR: $EXECUTABLE was not built"
    exit 1
  fi
  echo
  echo "=================  BUILD IS DONE ========"
fi

if [ "$1" != "buildonly" ] ; then
  if [ ! -f "$REPO_DIR/$EXECUTABLE" ] ; then
    echo "ERROR: $REPO_DIR/$EXECUTABLE not found"
    exit 1
  fi
  mkdir -p "$RUN_DIR/MPI_timings"
  cp "$REPO_DIR/$EXECUTABLE" "$RUN_DIR/"
  cp "$REPO_DIR/files/input" "$RUN_DIR/input"
  rm -f "$RUN_DIR/dims"

  cd "$RUN_DIR" || exit 1
  echo
  echo "=================  attempting mpirun  ========"
  _cmd="$OPENMPI_DIR/bin/mpirun -np $GESTS_NP $AOMP/bin/gpurun -rocmsmi ./$EXECUTABLE"
  echo "$_cmd"
  echo "np=$GESTS_NP threads=$OMP_NUM_THREADS"
  time $_cmd
  rc=$?
  if [ $rc != 0 ] ; then
    echo "ERROR: GESTS run failed"
    exit $rc
  fi
  echo
  echo "=================  DONE with mpirun  ========"
fi
