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

EPSDB_LIST=${EPSDB_LIST:-"examples smoke-limbo smoke omp5 openmpapps LLNL nekbone ovo sollve babelstream"}
SUITE_LIST=${SUITE_LIST:-"examples smoke-limbo smoke omp5 openmpapps LLNL nekbone ovo sollve"}
blockinglist="examples_fortran examples_openmp smoke nekbone sollve45 sollve50"

# Use bogus path to avoid using target.lst, a user-defined target list
# used by rocm_agent_enumerator.
export ROCM_TARGET_LST=/opt/nowhere

realpath=`realpath $0`
scriptdir=`dirname $realpath`
parentdir=`eval "cd $scriptdir;pwd;cd - > /dev/null"`
aompdir="$(dirname "$parentdir")"
resultsdir="$aompdir/bin/rocm-test/results"
scriptsdir="$aompdir/bin/rocm-test/scripts"
rocmtestdir="$aompdir"/bin/rocm-test
summary="$resultsdir"/summary.txt
unexpresults="$resultsdir"/unexpresults.txt
scriptfails=0
totalunexpectedfails=0

# make sure we see latest aomp dir
git pull
git log -1
EPSDB=1 ./clone_test.sh > /dev/null
AOMP_TEST_DIR=${AOMP_TEST_DIR:-"$HOME/git/aomp-test"}

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

$AOMP/bin/flang1 --version

clangversion=`$AOMP/bin/clang --version`
aomp=0
if [[ "$clangversion" =~ "AOMP_STANDALONE" ]]; then
  aomp=1
fi

# Parent dir should be ROCm base dir.
if [ $aomp -eq 1 ]; then
  AOMPROCM=$AOMP
else
  AOMPROCM=$AOMP/..
fi
export AOMPROCM
echo AOMPROCM=$AOMPROCM

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
if [ "$AOMP_GPU" == "" ]; then
  AOMP_GPU=$($AOMPROCM/bin/rocm_agent_enumerator | grep -m 1 -E gfx[^0]{1}.{2})
fi

# mygpu will eventually relocate to /opt/rocm/bin, support both cases for now.
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
echo AOMP_GPU=$AOMP_GPU
export AOMP_GPU

# Run quick sanity test
echo
echo "Helloworld sanity test:"
cd "$aompdir"/test/smoke/helloworld
make clean > /dev/null
OMP_TARGET_OFFLOAD=MANDATORY VERBOSE=1 make run > hello.log 2>&1
sed -n -e '/ld.lld/,$p' hello.log

# Determines ROCm version and openmp-extras version to choose which set
# of expected passes to use. Example ROCm 4.3 build installed but
# openmp-extras version is 4.2. This can happen in mainline builds. Do not expect
# this version mismatch on release testing. We will choose the lower version so that
# unsupported tests are not included.
function getversion(){
  supportedvers="4.3.0 4.4.0 4.5.0 4.5.2 5.0.0 5.1.0 5.2.0 5.3.0 5.4.3 5.5.0"
  declare -A versions
  versions[430]=4.3.0
  versions[440]=4.4.0
  versions[450]=4.5.0
  versions[452]=4.5.2
  versions[500]=5.0.0
  versions[510]=5.1.0
  versions[520]=5.2.0
  versions[530]=5.3.0
  versions[543]=5.4.3
  versions[550]=5.5.0

  if [ $aomp -eq 1 ]; then
    echo "AOMP detected at $AOMP, skipping ROCm version detections"
    maxvers=`echo $supportedvers | grep -o "[0-9].[0-9].[0-9]$" | sed -e 's/\.//g'`
    versionregex="(.*${versions[$maxvers]})"
    if [[ "$supportedvers" =~ $versionregex ]]; then
      finalvers=${BASH_REMATCH[1]}
    else
      echo "AOMP - Cannot select proper version list."
      exit 1
    fi
    echo "Selecting highest supported version: ${versions[$maxvers]}"
  else
    # Determine ROCm version.
    rocm=$(cat "$AOMPROCM"/.info/version*|head -1)
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
    ompextrasregex="openmp-extras-?[a-z]*-?\s*[0-9]+\.([0-9]+)\.([0-9]+)"
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
      echo "Using ompextrasver: $ompextrasver"
      compilerver=${versions[$ompextrasver]}
      initialver=$ompextrasver
    else
      echo "Using rocmver: $rocmver"
      compilerver=${versions[$rocmver]}
      initialver=$rocmver
    fi

    # There may be a patch release that is not in the supported list. To prevent
    # aggregation of all passing tests, attempt to choose the last supported
    # version with a major minor match. For example 5.4.4 may choose a passing list
    # for 5.4.3.
    if [ "$compilerver" == "" ]; then
      initialregex="([0-9][0-9])"
      [[ "$initialver" =~ $initialregex ]]
      majorminor=${BASH_REMATCH[1]}
      patchreleasever=`echo $supportedvers | sed -e 's|\.||g' | grep -m 2 -o "$majorminor[0-9]" | tail -1`
      compilerver=${versions[$patchreleasever]}
    fi

    if [ "$compilerver" == "" ]; then
      echo "Warning: Cannot detect compiler version or version is not supported in this script."
      echo "All expected passes were combined."
    fi

    echo Chosen Version: $compilerver
    versionregex="(.*$compilerver)"
    if [[ "$supportedvers" =~ $versionregex ]]; then
      finalvers=${BASH_REMATCH[1]}
    else
      echo "Error: Unsupported compiler build: $compilerver."
      exit 1
    fi
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
    cp passing-tests.txt "$resultsdir/$1"/"$1"_passing_tests.txt
  fi

  # Begin logging info in summary.txt.
  cd $resultsdir/$1
  if [ ! -f $summary ]; then
    echo "" > $summary
    echo "************************************" >> $summary
    echo "Detailed list of unexpected results:" >> $summary
  fi
  echo ===== $1 ===== | tee -a $summary $unexpresults

  # Sort expected passes
  for ver in $finalvers; do
    if [ -e "$rocmtestdir"/passes/$ver/$1/"$1"_passes.txt ]; then
      cat "$rocmtestdir"/passes/$ver/$1/"$1"_passes.txt >> "$1"_combined_exp_passes
    fi
  done
  sort -f -d "$1"_combined_exp_passes > "$1"_sorted_exp_passes
  passlines=`cat "$1"_sorted_exp_passes | wc -l`

  if [ -e "$1"_passing_tests.txt ]; then
    # Sort test reported passes
    sort -f -d "$1"_passing_tests.txt > "$1"_sorted_passes

    # Unexpected passes
    unexpectedpasses=$(diff "$1"_sorted_exp_passes "$1"_sorted_passes | grep '^>' | wc -l)
    echo Unexpected Passes: $unexpectedpasses | tee -a $summary $unexpresults
    diff "$1"_sorted_exp_passes "$1"_sorted_passes | grep '^>' | sed 's/> //' >> $summary
    echo >> $summary

    # Unexpected Fails
    if [ "$passlines" != 0 ]; then
      unexpectedfails=$(diff "$1"_sorted_exp_passes "$1"_sorted_passes | grep '^<' | wc -l)
    else
      unexpectedfails=0
    fi

    # Check unexpected fails for false negatives, i.e. tests that may have been deleted.
    if [ "$unexpectedfails" != 0 ]; then
      fails=`diff $1_sorted_exp_passes $1_sorted_passes | grep '^<' | sed "s|< ||g"`
      if [[ "$1" =~ examples|smoke|omp5 ]]; then
        # For smoke, examples, and omp5 the missing test will have no directory or the directory is missing a Makefile.
        # This can happen if there is a test binary that is not cleaned up, which keeps the test directory present.
        if [ -e "$resultsdir/$1/$1_make_fail.txt" ]; then
          for fail in $fails; do
            notpresent=0
            if [ ! -d "$2/$fail" ]; then
              notpresent=1
            else
              pushd "$2/$fail" > /dev/null
              # If no Makefile then assume this is a recently deleted test.
              if [ ! -e Makefile ]; then
                notpresent=1
              fi
              popd > /dev/null
            fi
            if [ "$notpresent" == 1 ]; then
              warnings[$1]+="$fail, "
              ((unexpectedfails--))
              ((warningcount++))
            fi
          done
        fi
      elif [[ "$1" =~ sollve|ovo|LLNL|openmpapps ]]; then
        # Combine passing/failing tests, which shows all tests that tried to build/run.
        # If the unexpected failure is not on that list, warn the user that test may be missing
        # from suite.
        if [ -e "$1"_failing_tests.txt ]; then
          cat "$1"_failing_tests.txt | tee -a "$resultsdir"/"$1"/"$1"_all_tests.txt
        fi
        if [ -e "$1"_make_fail.txt ]; then
          cat "$1"_make_fail.txt | tee -a "$resultsdir"/"$1"/"$1"_all_tests.txt
        fi
        if [ -e "$1"_passing_tests.txt ]; then
          cat "$1"_passing_tests.txt | tee -a "$resultsdir"/"$1"/"$1"_all_tests.txt
        fi
        if [ -e "$resultsdir"/"$1"/"$1"_all_tests.txt ]; then
          for fail in $fails; do
            match=`grep -e "^$fail$" "$resultsdir"/"$1"/"$1"_all_tests.txt`
            # No match means test was possibly removed
            if [ "$match" == "" ]; then
              warnings[$1]+="$fail, "
              ((unexpectedfails--))
              ((warningcount++))
            fi
          done
        fi
      fi
    fi # End unexpected fail parsing for missing tests
    if [ "$EPSDB" == "1" ]; then
      for suite in $blockinglist; do
        if [ "$1" == "$suite" ]; then
          ((totalunexpectedfails+=$unexpectedfails))
          break
        fi
      done
    else
      ((totalunexpectedfails+=$unexpectedfails))
    fi

    echo "Unexpected Fails: $unexpectedfails" | tee -a $summary $unexpresults
    diff "$1"_sorted_exp_passes "$1"_sorted_passes | grep '^<' | sed 's/< //' >> $summary
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
    echo "Unexpected Passes: 0" | tee -a $summary $unexpresults
    if [ "$passlines" != 0 ]; then
      numtests=$(cat "$resultsdir"/"$1"/"$1"_sorted_exp_passes | wc -l)
    else
      numtests=0
    fi
    echo "Unexpected Fails: $numtests" | tee -a $summary $unexpresults
    cat "$1"_sorted_exp_passes >> $summary
    if [ "$EPSDB" == "1" ]; then
      for suite in $blockinglist; do
        if [ "$1" == "$suite" ]; then
          ((totalunexpectedfails+=$numtests))
          break
        fi
      done
    else
      ((totalunexpectedfails+=$numtests))
    fi
  fi
}

function checkrc(){
  if [ "$1" != 0 ]; then
    ((scriptfails++))
  fi
}


# Get version of installed ROCm and compare against openmp-extras version.
getversion
echo "Included Versions: $finalvers"

function examples(){
  # Fortran Examples
  mkdir -p "$resultsdir"/examples_fortran
  #echo "cp -rf "$AOMP"/examples/fortran "$aompdir"/examples"
  #cp -rf "$AOMP"/examples/fortran "$aompdir"/examples
  cd "$aompdir"/examples/fortran
  EPSDB=1 AOMPHIP=$AOMPROCM ../check_examples.sh fortran
  checkrc $?
  copyresults examples_fortran "$aompdir"/examples/fortran

  # Openmp Examples
  mkdir -p "$resultsdir"/examples_openmp
  #echo "cp -rf "$AOMP"/examples/openmp "$aompdir"/examples"
  #cp -rf "$AOMP"/examples/openmp "$aompdir"/examples
  cd "$aompdir"/examples/openmp
  EPSDB=1 ../check_examples.sh openmp
  checkrc $?
  copyresults examples_openmp "$aompdir"/examples/openmp
}

function smoke(){
  # Smoke
  mkdir -p "$resultsdir"/smoke
  cd "$aompdir"/test/smoke
  AOMP_PARALLEL_SMOKE=1 CLEANUP=0 AOMPHIP=$AOMPROCM ./check_smoke.sh
  checkrc $?
  copyresults smoke
}

SMOKE_FAILS=${SMOKE_FAILS:-1}
function smokefails(){
  # Smoke-fails
  if [ "$SMOKE_FAILS" == "1" ]; then
    mkdir -p "$resultsdir"/smoke-fails
    cd "$aompdir"/test/smoke-fails
    ./check_smoke_fails.sh
    checkrc $?
    copyresults smoke-fails
  else
    echo "Skipping smoke-fails."
  fi
}

SMOKE_LIMBO=${SMOKE_LIMBO:-1}
function smoke-limbo(){
  # Smoke-fails
  if [ "$SMOKE_LIMBO" == "1" ]; then
    mkdir -p "$resultsdir"/smoke-limbo
    cd "$aompdir"/test/smoke-limbo
    ./check_smoke_limbo.sh
    checkrc $?
    copyresults smoke-limbo
  else
    echo "Skipping smoke-limbo."
  fi
}

function omp5(){
  # Omp5
  mkdir -p "$resultsdir"/omp5
  cd "$aompdir"/test/omp5
  ./check_omp5.sh
  checkrc $?
  copyresults omp5 "$aompdir"/test/omp5
}

function openmpapps(){
  # -----Run Openmpapps-----
  mkdir -p "$resultsdir"/openmpapps
  cd "$AOMP_TEST_DIR"/openmpapps
  ./check_openmpapps.sh
  copyresults openmpapps
}

function nekbone(){
  # -----Run Nekbone-----
  mkdir -p "$resultsdir"/nekbone
  cd "$aompdir"/bin
  VERBOSE=0 ./run_nekbone.sh
  cd "$AOMP_TEST_DIR"/Nekbone/test/nek_gpu1
  copyresults nekbone
}

function sollve(){
  # Sollve
  mkdir -p "$resultsdir"/sollve45
  mkdir -p "$resultsdir"/sollve50
  cd "$aompdir"/bin

  no_usm_gpus="gfx900 gfx906"
  if [[ "$no_usm_gpus" =~ "$AOMP_GPU" ]]; then
    echo "Skipping USM 5.0 tests."
    SKIP_USM=1 SKIP_SOLLVE51=1 ./run_sollve.sh
  else
    SKIP_SOLLVE51=1 ./run_sollve.sh
  fi

  ./check_sollve.sh
  checkrc $?

  # 4.5 Results
  cd "$HOME"/git/aomp-test/sollve_vv/results_report45
  copyresults sollve45

  # 5.0 Results
  cd "$HOME"/git/aomp-test/sollve_vv/results_report50
  copyresults sollve50
}

function babelstream(){
  mkdir -p "$resultsdir"/babelstream
  cd "$aompdir"/bin
  if [ $aomp -eq 0 ]; then
    export ROCMINFO_BINARY=$AOMP/../bin/rocminfo
  fi
  ./run_babelstream.sh
  cd "$AOMP_TEST_DIR"/babelstream
  copyresults babelstream
}

function LLNL(){
  mkdir -p "$resultsdir"/LLNL
  cd "$aompdir"/test/LLNL/openmp5.0-tests
  ./check_LLNL.sh log
  "$scriptsdir"/parse_LLNL.sh
  copyresults LLNL
}

function ovo(){
  mkdir -p "$resultsdir"/ovo
  cd "$aompdir"/bin
  ./run_ovo.sh log "$HOME"/git/aomp-test/OvO
  "$scriptsdir"/parse_OvO.sh
  cd "$AOMP_TEST_DIR"/OvO
  copyresults ovo
}

# Clean Results
cd "$aompdir"/bin
rm -rf $resultsdir
mkdir -p $resultsdir

# Run Tests
if [ "$EPSDB" == "1" ]; then
  SUITE_LIST="$EPSDB_LIST"
fi
echo Running List: $SUITE_LIST

declare -A warnings
warningcount=0
for suite in $SUITE_LIST; do
  $suite
done

echo "************************************" >> $summary

if [ "$compilerver" == "" ]; then
  echo "Warning: Cannot detect compiler version or version is not supported in this script." >> $summary
  echo "All expected passes were combined." >> $summary
fi

echo "" >> $summary
echo "Condensed Summary:" >> $summary
if [ -f $unexpresults ]; then
  cat $unexpresults >> $summary
fi

# Print warnings for possible missing tests.
if [ ${#warnings[@]} != 0 ]; then
  echo "" >> $summary
  echo "--------------------------- MISSING TEST WARNINGS ---------------------------" >> $summary
  echo "These tests may no longer exist in their respective suite, but are still present in expected passes." >> $summary
  for i in "${!warnings[@]}"; do
    val=${warnings[$i]}
    echo "$i: $val" >> $summary
    echo "" >> $summary
  done
  echo "----------------------------------------------------------------" >> $summary
fi

echo >> $summary
echo Overall Unexpected fails: $totalunexpectedfails >> $summary
echo Script Errors: $scriptfails >> $summary
echo Test Warnings: $warningcount >> $summary
if [ "$totalunexpectedfails" -gt 0 ] || [ "$scriptfails" != 0 ]; then
  echo FAIL >> $summary
  echo "EPSDB Status:  red" >> $summary
else
  echo PASS >> $summary
  echo "EPSDB Status:  green" >> $summary
fi

echo ""
echo >> $summary
cat $summary
exit $totalunexpectedfails
