#!/bin/bash
#
#  run_rocm_test.sh
#
#  WARNING: This is intended for verifying ROCm builds and is not meant for
#  usage with AOMP standalone builds.
#
#  Please check with Ron or Ethan for script modifications.
#
#

# Use bogus path to avoid using target.lst, a user-defined target list
# used by rocm_agent_enumerator.
export ROCM_TARGET_LST=/opt/nowhere

scriptdir=$(dirname "$0")
parentdir=`eval "cd $scriptdir;pwd;cd - > /dev/null"`
aompdir="$(dirname "$parentdir")"
resultsdir="$aompdir/bin/rocm-test/results"
rocmtestdir="$aompdir"/bin/rocm-test
summary="$resultsdir"/summary.txt
unexpresults="$resultsdir"/unexpresults.txt
scriptfails=0

# Set AOMP to point to rocm symlink or newest version.
if [ -L /opt/rocm ]; then
  AOMP=${AOMP:-"/opt/rocm/llvm"}
else
  newestrocm=$(ls --sort=time /opt | grep -m 1 rocm)
  AOMP=${AOMP:-"/opt/$newestrocm/llvm"}
fi
export AOMP
echo "AOMP = $AOMP"

# Make sure clang is present.
$AOMP/bin/clang --version
if [ $? -ne 0 ]; then
  echo "Error: Clang not found at "$AOMP"/bin/clang."
  exit 1
fi

# Parent dir should be ROCm base dir.
AOMPROCM=$AOMP/..
export AOMPROCM
echo AOMPROCM= $AOMPROCM

# Set ROCM_LLVM for examples
export ROCM_LLVM=$AOMP

#unset ROCM_PATH

# Use bogus path to avoid using target.lst, a user-defined target list
# used by rocm_agent_enumerator.
export ROCM_TARGET_LST=/opt/nowhere

echo "RAE devices:"
$AOMPROCM/bin/rocm_agent_enumerator

# Set AOMP_GPU.
# Regex skips first result 'gfx000' and selects second id.
AOMP_GPU=$($AOMPROCM/bin/rocm_agent_enumerator | grep -m 1 -E gfx[^0]{1}.{2})
# mygpu will eventually relocate to /opt/rocm/bin, support both cases for now.
echo AOMP_GPU= $AOMP_GPU

if [ "$AOMP_GPU" != "" ]; then
  echo "AOMP_GPU set with rocm_agent_enumerator."
else
  echo "AOMP_GPU is empty, use mygpu."
  if [ -a $AOMP/bin/mygpu ]; then
    AOMP_GPU=$($AOMP/bin/mygpu)
  else
    AOMP_GPU=$($AOMP/../bin/mygpu)
  fi
fi
if [ "$AOMP_GPU" == "" ]; then
  echo "Error: AOMP_GPU was not able to be set with RAE or mygpu."
  exit 1
fi
export AOMP_GPU

# Determines ROCm version and openmp-extras version to choose which set
# of expected passes to use. Example ROCm 4.3 build installed but
# openmp-extras version is 4.2. This can happen in mainline builds. Do not expect
# this version mismatch on release testing. We will choose the lower version so that
# unsupported tests are not included.
function getversion(){
  supportedvers="4.3.0 4.4.0 4.5.0 4.5.2 5.0.0"
  declare -A versions
  versions[430]=4.3.0
  versions[440]=4.4.0
  versions[450]=4.5.0
  versions[452]=4.5.2
  versions[500]=5.0.0

  # Determine ROCm version.
  rocm=$(cat "$AOMPROCM"/.info/version-dev)
  rocmregex="([0-9]+\.[0-9]+\.[0-9]+)"
  if [[ "$rocm" =~ $rocmregex ]]; then
    rocmver=$(echo ${BASH_REMATCH[1]} | sed "s/\.//g")
    echo rocmver: $rocmver
  else
    echo Unable to determine rocm version.
    exit 1
  fi

  # Determine OS flavor to properly query openmp-extras version.
  osname=$(cat /etc/os-release | grep -e ^NAME=)
  # Regex to cover single/multi version installs for deb/rpm.
  ompextrasregex="openmp-extras[0-9]*\.*[0-9]*\.*[0-9]*-*\s*[0-9]+\.([0-9]+)\.([0-9]+)"
  rpmregex="Red Hat|CentOS|SLES"
  echo $osname
  if [[ "$osname" =~ $rpmregex ]]; then
    echo "Red Hat/CentOS/SLES found"
    ompextraspkg=$(rpm -qa | grep openmp-extras | tail -1)
  elif [[ $osname =~ "Ubuntu" ]]; then
    echo "Ubuntu found"
    ompextraspkg=$(dpkg --list | grep openmp-extras | tail -1)
  fi
  if [[ "$ompextraspkg" =~ $ompextrasregex ]]; then
    ompextrasver=${BASH_REMATCH[1]}${BASH_REMATCH[2]}
    echo ompextrasver: $ompextrasver
  else
    echo Unable to determine openmp-extras package version.
    exit 1
  fi
  # Set the final version to use for expected passing lists. The expected passes
  # will include an aggregation of suported versions up to and including the chosen
  # version.  Example: If 4.4 is selected then the final list will include expected passes
  # from 4.3 and 4.4. Openmp-extras should not be a higher version than rocm.
  if [ "$rocmver" == "$ompextrasver" ] || [ "$rocmver" -gt "$ompextrasver" ]; then
    compilerver=${versions[$ompextrasver]}
  else
    compilerver=${versions[$rocmver]}
  fi
  echo Chosen Version: $compilerver
  versionregex="(.*$compilerver)"
  if [[ "$supportedvers" =~ $versionregex ]]; then
    finalvers=${BASH_REMATCH[1]}
  else
    echo "Unsupported compiler build."
    exit 1
  fi
}

function copyresults(){
  # $1 name of test suite

  # Copy logs from suite to results folder
  if [ -e failing-tests.txt ]; then
    cp failing-tests.txt "$resultsdir/$1"/"$1"_failing_tests.txt
  fi
  if [ -e make-fail.txt ]; then
    cp make-fail.txt "$resultsdir/$1"/"$1"_make_fail.txt
  fi
  if [ -e passing-tests.txt ]; then
    echo""
    cp passing-tests.txt "$resultsdir/$1"/"$1"_passing_tests.txt
  fi


  # Begin logging info in summary.txt.
  cd $resultsdir/$1
  echo ===== $1 ===== | tee -a $summary $unexpresults
  if [ -e "$1"_passing_tests.txt ]; then
    # Sort test reported passes
    sort -f -d "$1"_passing_tests.txt > "$1"_sorted_passes

    # Sort expected passes
    for ver in $finalvers; do
      if [ -e "$rocmtestdir"/passes/$ver/$1/"$1"_passes.txt ]; then
        cat "$rocmtestdir"/passes/$ver/$1/"$1"_passes.txt >> "$1"_combined_exp_passes
      fi
    done
    sort -f -d "$1"_combined_exp_passes > "$1"_sorted_exp_passes

    # Unexpected passes
    unexpectedpasses=$(diff "$1"_sorted_exp_passes "$1"_sorted_passes | grep '>' | wc -l)
    echo Unexpected Passes: $unexpectedpasses | tee -a $summary $unexpresults
    diff "$1"_sorted_exp_passes "$1"_sorted_passes | grep '>' | sed 's/> //' >> $summary
    echo >> $summary

    # Unexpected Fails
    unexpectedfails=$(diff "$1"_sorted_exp_passes "$1"_sorted_passes | grep '<' | wc -l)
    ((totalunexpectedfails+=$unexpectedfails))
    echo "Unexpected Fails: $unexpectedfails" | tee -a $summary $unexpresults
    diff "$1"_sorted_exp_passes "$1"_sorted_passes | grep '<' | sed 's/< //' >> $summary
    echo >> $summary

    # Failing Tests
    if [ -e "$1"_failing_tests.txt ]; then
      echo Runtime Fails: >> $summary
      cat "$1"_failing_tests.txt >> $summary
      echo >> $summary
    fi
    if [ -e "$1"_make_fail.txt ]; then
      echo Compile Fails: >> $summary
      cat "$1"_make_fail.txt >> $summary
      echo >> $summary
    fi

  else
    # No passing-tests.txt found, count expected passes as fails.
    # Does not count combined expected passes from previous rocm versions
    numtests=$(cat "$rocmtestdir"/passes/"$compilerver"/"$1"/"$1"_passes.txt | wc -l)
    ((totalunexpectedfails+=$numtests))
  fi
}

function checkrc(){
  if [ "$1" != 0 ]; then
    ((scriptfails++))
  fi
}

function setUnsetAsanEnv(){
  set CLANG_VERSION=$($AOMP/bin/clang --version | head -n 1 | grep -om 1 "[0-9]\+\.[0-9]\.[0-9]")
  flag="$1"
  case $flag in
          true)
              export LD_PRELOAD="$AOMP/lib/clang/$CLANG_VERSION/lib/linux/libclang_rt.asan-x86_64.so"
              export ASAN_OPTIONS="verbosity=2:symbolize=1:detect_leaks=0"
              ;;
          false)
              unset LD_PRELOAD ASAN_OPTIONS
              ;;
  esac
}

# Get version of installed ROCm and compare against openmp-extras version.
getversion
echo "Included Versions: $finalvers"

if [ ! -z "$1" ] && [ "$1" == "asan" ]; then
        setUnsetAsanEnv true
fi

cd "$aompdir"/bin
rm -rf $resultsdir
mkdir -p $resultsdir

# Fortran Examples
mkdir -p "$resultsdir"/fortran
#echo "cp -rf "$AOMP"/examples/fortran "$aompdir"/examples"
#cp -rf "$AOMP"/examples/fortran "$aompdir"/examples
cd "$aompdir"/examples/fortran
EPSDB=1 AOMPHIP=$AOMPROCM ../check_examples.sh fortran
checkrc $?
copyresults fortran

# Openmp Examples
mkdir -p "$resultsdir"/openmp
#echo "cp -rf "$AOMP"/examples/openmp "$aompdir"/examples"
#cp -rf "$AOMP"/examples/openmp "$aompdir"/examples
cd "$aompdir"/examples/openmp
EPSDB=1 ../check_examples.sh openmp
checkrc $?
copyresults openmp

# Smoke
mkdir -p "$resultsdir"/smoke
cd "$aompdir"/test/smoke
CLEANUP=0 AOMPHIP=$AOMPROCM ./check_smoke.sh
checkrc $?
copyresults smoke

# Smoke-fails
mkdir -p "$resultsdir"/smoke-fails
cd "$aompdir"/test/smoke-fails
./check_smoke_fails.sh
checkrc $?
copyresults smoke-fails

# Omp5
mkdir -p "$resultsdir"/omp5
cd "$aompdir"/test/omp5
./check_omp5.sh
checkrc $?
copyresults omp5


# Sollve
mkdir -p "$resultsdir"/sollve45
mkdir -p "$resultsdir"/sollve50
cd "$aompdir"/bin
./clone_aomp_test.sh
./run_sollve.sh
./check_sollve.sh
checkrc $?

# 4.5 Results
cd "$HOME"/git/aomp-test/sollve_vv/results_report45
copyresults sollve45

# 5.0 Results
cd "$HOME"/git/aomp-test/sollve_vv/results_report50
copyresults sollve50

if [ ! -z "$1" ] && [ "$1" == "asan" ]; then
        setUnsetAsanEnv false
fi

cat $unexpresults >> $summary
echo >> $summary
echo Overall Unexpected fails: $totalunexpectedfails >> $summary
echo Script Errors: $scriptfails >> $summary
if [ "$totalunexpectedfails" -gt 0 ] || [ "$scriptfails" != 0 ]; then
  echo FAIL >> $summary
else
  echo PASS >> $summary
fi

cat $summary
