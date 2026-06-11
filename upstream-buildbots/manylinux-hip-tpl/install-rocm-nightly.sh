#!/bin/bash
# shellcheck shell=bash
#
# Install a TheRock ROCm "dist" nightly tarball into a target directory.
#
# The full dist tarball contains the complete ROCm SDK (HIP runtime, device
# libs, rocminfo, rocblas, rocthrust, ...), so it fully replaces the previous
# yum/rocm.repo based install used by the manylinux buildbot images.
#
# Inputs are read from the environment so the Dockerfile can wire them through
# build ARGs:
#   ROCM_BASE_VERSION    base ROCm version, e.g. "7.14"        (default 7.14)
#   ROCM_GFX             gfx target family, e.g. "gfx90a"      (default gfx90a)
#   ROCM_NIGHTLY_DATE    pin a build date YYYYMMDD; empty =>   (default empty)
#                        auto-detect the most recent nightly

set -euo pipefail

RocmBaseVersion="${ROCM_BASE_VERSION:-7.14}"
RocmGfx="${ROCM_GFX:-gfx90a}"
RocmNightlyDate="${ROCM_NIGHTLY_DATE:-}"
RocmInstallDir="/opt/rocm"
RocmNightlyBaseUrl="https://rocm.nightlies.amd.com/tarball"

if [ -n "${RocmNightlyDate}" ] && [[ ! "${RocmNightlyDate}" =~ ^[0-9]{8}$ ]]; then
  echo "error: ROCM_NIGHTLY_DATE must be an 8-digit date (YYYYMMDD), got '${RocmNightlyDate}'" >&2
  exit 1
fi

# Shared curl options: fail on HTTP errors, stay quiet but show errors, follow
# redirects, bound the connection setup, and retry transient failures. The
# per-call --max-time bounds the whole transfer and is set at each call site
# because the index fetch and the multi-GB tarball download need very different
# ceilings.
CurlOpts=(--fail --silent --show-error --location --connect-timeout 30 --retry 5 --retry-delay 5)

# Resolve the tarball filename, either from a pinned date or by querying the
# nightly index for the most recent build matching version + gfx.
function resolveTarball() {
  local Prefix="therock-dist-linux-${RocmGfx}-${RocmBaseVersion}.0a"

  if [ -n "${RocmNightlyDate}" ]; then
    echo "${Prefix}${RocmNightlyDate}.tar.gz"
    return 0
  fi

  # The index page embeds the available files; extract every name matching our
  # version+gfx, sort lexically (dates are zero-padded YYYYMMDD), take newest.
  local Latest
  Latest="$(curl "${CurlOpts[@]}" --max-time 60 "${RocmNightlyBaseUrl}/" |
    grep -oP "${Prefix}[0-9]{8}\.tar\.gz" |
    sort -u |
    tail -1)"

  if [ -z "${Latest}" ]; then
    echo "error: no nightly tarball found for ${RocmGfx} version ${RocmBaseVersion} at ${RocmNightlyBaseUrl}/" >&2
    return 1
  fi

  echo "${Latest}"
}

function doInstall() {
  local Tarball
  Tarball="$(resolveTarball)"
  local Url="${RocmNightlyBaseUrl}/${Tarball}"

  echo "Installing ROCm nightly: ${Tarball}"
  echo "  from: ${Url}"
  echo "  into: ${RocmInstallDir}"

  local TmpDir
  TmpDir="$(mktemp -d)"
  # shellcheck disable=SC2064
  trap "rm -rf '${TmpDir}'" EXIT

  # The tarball can be multiple GB; allow up to 30 minutes for the transfer.
  curl "${CurlOpts[@]}" --max-time 1800 -o "${TmpDir}/${Tarball}" "${Url}"
  mkdir -p "${RocmInstallDir}"
  tar -xf "${TmpDir}/${Tarball}" -C "${RocmInstallDir}"

  # Record what was actually installed. TheRock's own /opt/rocm/.info/version
  # only carries the base version (e.g. 7.14.0), which cannot distinguish
  # nightlies. The trailing 8 digits of the tarball name are the build date.
  local Date="${Tarball: -15:8}"
  local InfoDir="${RocmInstallDir}/.info"
  if [ ! -d "${InfoDir}" ]; then
    echo "error: expected ${InfoDir} to exist after extraction; the tarball may be incomplete or have an unexpected layout" >&2
    return 1
  fi
  {
    echo "tarball=${Tarball}"
    echo "url=${Url}"
    echo "base_version=${RocmBaseVersion}"
    echo "gfx=${RocmGfx}"
    echo "date=${Date}"
  } >"${InfoDir}/nightly"

  echo "ROCm nightly install complete."
}

doInstall
