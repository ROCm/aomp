#!/usr/bin/env bash
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Tests for the directory locks that keep two runs out of one build tree,
# checkout or interpreter. The interpreter can sit on a filesystem shared
# between machines, so the cases that matter most are the ones about an owner
# this machine cannot see: a lock recorded by another host must be waited for,
# never reclaimed.
#
# Run: bash bin/aux/rocke/tests/test_locks.sh

# shellcheck source-path=SCRIPTDIR
set -u

Here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
Passed=0
Failed=0

# What the driver owns and the module reads.
HeldLockDirs=()
HostName="$(uname -n)"
ROCKE_LOCK_WAIT=2

function failSetup { echo "refused: ${1}"; exit 3; }

# shellcheck source=../rocke_locks.source
. "${Here}/../rocke_locks.source"

function check {  # <description> <expected> <actual>
  if [[ "${2}" == "${3}" ]]; then
    (( ++Passed ))
  else
    (( ++Failed ))
    printf 'FAIL  %s\n      expected [%s]\n      got      [%s]\n' \
      "${1}" "${2}" "${3}"
  fi
}

# Each case gets its own directory, so one leftover lock cannot decide another.
function lockPath {
  printf '%s/lock.d' "$(mktemp -d)"
}

Lock="$(lockPath)"
acquireDirLock "${Lock}" test
check "acquiring records this host and process" \
  "${HostName} $$" "$(< "${Lock}/owner")"
check "the held lock is remembered for the exit path" \
  "${Lock}" "${HeldLockDirs[*]}"

releaseDirLock "${Lock}"
check "releasing removes the lock" "gone" \
  "$([[ -e "${Lock}" ]] && echo present || echo gone)"
check "releasing forgets it" "" "${HeldLockDirs[*]}"

# A lock left by a process of this host that no longer exists is reclaimable:
# pid 1 is init, so a pid that cannot exist stands in for a dead one.
Lock="$(lockPath)"
mkdir -p "${Lock}"
printf '%s 999999999\n' "${HostName}" > "${Lock}/owner"
acquireDirLock "${Lock}" test
check "a lock left by a dead local process is reclaimed" \
  "${HostName} $$" "$(< "${Lock}/owner")"
releaseDirLock "${Lock}"

# The case the shared interpreter depends on: this process table says nothing
# about a process on another machine, so the lock has to be left alone.
Lock="$(lockPath)"
mkdir -p "${Lock}"
printf 'another-host 999999999\n' > "${Lock}/owner"
Out="$( (acquireDirLock "${Lock}" test) 2>&1 | tail -1 )"
Want="refused: gave up after ${ROCKE_LOCK_WAIT}s waiting for the test lock:"
Want+=" ${Lock}"
check "a lock held by another host is waited for, not reclaimed" \
  "${Want}" "${Out}"
check "and it still belongs to that host" \
  "another-host 999999999" "$(< "${Lock}/owner")"

# An owner record that cannot be read at all is refused rather than guessed.
Lock="$(lockPath)"
mkdir -p "${Lock}"
printf 'nonsense\n' > "${Lock}/owner"
Out="$( (acquireDirLock "${Lock}" test) 2>&1 | tail -1 )"
check "a malformed owner record is refused" \
  "refused: malformed test lock owner: ${Lock}" "${Out}"

# A symlinked lock would let a run write outside the tree it locked.
Lock="$(lockPath)"
ln -s /tmp "${Lock}"
Out="$( (acquireDirLock "${Lock}" test) 2>&1 | tail -1 )"
check "a symlinked lock is refused" \
  "refused: refusing symlinked test lock: ${Lock}" "${Out}"

# Releasing is only for the owner: another holder's lock stays put.
Lock="$(lockPath)"
mkdir -p "${Lock}"
printf 'another-host 999999999\n' > "${Lock}/owner"
releaseDirLock "${Lock}"
check "releasing leaves a lock this process does not own" \
  "present" "$([[ -e "${Lock}" ]] && echo present || echo gone)"

echo "${Passed} passed, ${Failed} failed"
(( Failed == 0 ))
