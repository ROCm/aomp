# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Resolving the compiler under test, and proving every tool comes from it.
#
# Sourced by ../../run_rocke.sh, which owns the run: this file defines functions
# and nothing else, so sourcing it cannot change state or fail a run on its own.
# shellcheck shell=bash
#
# The driver owns the run's state -- RocmRoot, PyBin, BuildRoot, Stage,
# LaneRelevance and the rest -- and these functions read it without ever
# assigning it. Checked on its own, shellcheck cannot see where that state
# comes from, so SC2154 is off for the file; check the driver too, since -x
# follows a source for definitions but reports nothing inside it.
# shellcheck disable=SC2154

# Walk up from the resolved llvm dir to the nearest ancestor shipping comgr or
# hipcc. Key on those, not on include/hip or amdgcn/bitcode: some packagings put
# headers and device-libs under llvm/, so keying on those would stop one level
# below the real root and let comgr fall back to a system /opt/rocm.
function resolveRocmRoot {
  local Dir="${1}" Start="${1}" _Hop
  for _Hop in 0 1 2 3; do
    if [[ -e "${Dir}/lib/libamd_comgr.so" || -x "${Dir}/bin/hipcc" ]]; then
      echo "${Dir}"; return 0
    fi
    Dir="$(realpath -m "${Dir}/..")"
  done
  realpath -m "$(dirname "${Start}")"  # give up; prefix check + hygiene flag it
}

# Populate the shim named in PATH above. Separate from the PATH line because it
# needs fatalSetup, and definitions in this file come after that line runs.
function installCodShim {
  if ! mkdir -p "${CodShim}" \
    || ! ln -sf "${AOMP}/bin/clang++" "${CodShim}/c++" \
    || ! ln -sf "${AOMP}/bin/clang" "${CodShim}/cc"; then
    fatalSetup "cannot create the COD compiler shim in ${CodShim}" harness
  fi
}

function printBanner {
  local ClangVer LlvmSha HipVer
  ClangVer="$("${CXX}" --version 2>/dev/null | head -1)"
  # The COD clang embeds its llvm-project git SHA in --version; grab it so a stale
  # COD (or a re-tagged same-SHA build) is identifiable from the log alone.
  LlvmSha="$("${CXX}" --version 2>/dev/null | grep -oE '[0-9a-f]{12,40}' | tail -1)"
  HipVer="$(awk -F= '
    /^HIP_VERSION_(MAJOR|MINOR|PATCH|GITHASH)=/ { v[$1] = $2 }
    END { if (v["HIP_VERSION_MAJOR"] != "")
            printf "%s.%s.%s-%s", v["HIP_VERSION_MAJOR"], v["HIP_VERSION_MINOR"], \
                                  v["HIP_VERSION_PATCH"], v["HIP_VERSION_GITHASH"] }
  ' "${RocmRoot}/share/hip/version" 2>/dev/null)"
  echo "==============================================================================="
  echo "rocKE ${Stage}  ($(date '+%Y-%m-%d %H:%M:%S'))"
  echo "  AOMP        = ${AompInput} -> ${AOMP}"
  echo "  ROCM_PATH   = ${ROCM_PATH} (${RocmRootSource})"
  echo "  clang       = ${ClangVer}"
  echo "  llvm SHA    = ${LlvmSha:-?}"
  echo "  HIP         = ${HipVer:-?}"
  echo "  flavors     = codegen:${ROCKE_CODEGEN_FLAVOR}  comgr:${ROCKE_COMGR_FLAVOR}  engine:${ROCKE_ENGINE_FLAVORS}"
  echo "==============================================================================="
}

# True when a resolved path lives inside the COD install root.
function underCod {
  [[ -n "${1}" && -e "${1}" && "$(realpath -m "${1}")" == "${RocmRoot}"/* ]]
}

# Print one hygiene row; return non-zero when a *hard* requirement is external.
function codToolchainRow {  # <label> <path> <hard:1|0>
  local Label="${1}" Path="${2}" Hard="${3}" Tag="MISSING"
  if [[ -n "${Path}" && -e "${Path}" ]]; then
    Tag="EXTERNAL"
    underCod "${Path}" && Tag="COD"
  fi
  printf '  %-13s [%-8s] %s\n' "${Label}" "${Tag}" "${Path:-<not found>}"
  [[ "${Hard}" == 1 && "${Tag}" != COD ]] && return 1
  return 0
}

# Datalayout generation the COD clang itself emits, from its target datalayout p8
# field. This cannot name a flavor: rocKE's llvm22 and llvm23 share the indexed p8
# shape, so the field distinguishes generations, not releases. It is still the one
# signal that cannot leak from an unrelated tree, which makes it the right
# cross-check against the ROCm number the comgr reports.
function codClangP8Shape {
  local Arch="${1}" Dl
  Dl=$(printf 'int _rocke_flavor_probe;\n' | "${AOMP}/bin/clang" -x c \
        -target amdgcn-amd-amdhsa -mcpu="${Arch}" -emit-llvm -S - -o - 2>/dev/null \
        | sed -n 's/^target datalayout = "\(.*\)"/\1/p')
  case "${Dl}" in
    *p8:128:128:128:48*) echo indexed ;;
    *p8:128:128-*)       echo plain ;;
    *)                   echo unknown ;;
  esac
}

# Pin one flavor knob to the COD clang's own flavor when it is 'auto'; warn when
# an explicit value disagrees, since rocKE would then lower IR in the wrong one.
function resolveFlavorKnob {  # <env-var-name> <flavor>
  local Name="${1}" Flavor="${2}" Cur="${!1}"
  if [[ "${Cur}" == auto ]]; then
    [[ "${Flavor}" != "?" && -n "${Flavor}" ]] \
      || fatalSetup "cannot determine the IR flavor for ${Name}" toolchain
    export "${Name}=${Flavor}"
  elif [[ "${Flavor}" != "?" && "${Cur}" != "${Flavor}" ]]; then
    echo "WARNING: ${Name}=${Cur} overrides the flavor this COD implies (${Flavor})"
  fi
}

# Prove the compiler-critical tools/libs resolve *inside* the COD install, so a
# green row can never come from a stale system ROCm. clang/comgr are always hard
# requirements; HIP/hipcc/llvm-readelf are hard only in the lanes that use them.
function assertCodToolchain {
  local Probe Comgr ComgrVer ComgrFlavor ComgrIface ClangShape Pin Rc=0
  local HipHard=0 HipccHard=0 ReadelfHard=0 CxxHard=0 ComgrVersionTrusted=1
  # rocke_cod_probe.py reports the comgr lib rocKE will actually load, its ROCm
  # vintage, the IR flavor *rocKE derives* from that vintage, and the lib's own
  # interface version. The flavor comes from rocKE's own ladder, so a release that
  # adds a flavor needs no edit here.
  ClangShape="$(codClangP8Shape "${ROCKE_CI_ARCHES%% *}")"
  # The clang datalayout is the one basis for the flavor that cannot leak in from an
  # unrelated tree. Losing it means the flavor falls back to the comgr's ROCm number,
  # which is exactly the value this gate spends thirty lines distrusting -- so say so
  # in a row instead of continuing on the weaker basis in silence. A new p8 shape
  # upstream lands here, and that is worth a night's attention.
  if [[ "${ClangShape}" == unknown ]]; then
    rockeResult setup cod-datalayout 1 \
      "the COD clang emits a p8 datalayout rocKE does not describe: pinning the flavor from the comgr's ROCm number instead" \
      harness
  fi
  Probe="$("${PyBin}" "${HelperDir}/rocke_cod_probe.py" "${ClangShape}" 2>/dev/null)"
  read -r Pin ComgrFlavor ComgrVer ComgrIface Comgr <<< "${Probe}"
  resolveFlavorKnob ROCKE_CODEGEN_FLAVOR "${Pin}"
  resolveFlavorKnob ROCKE_COMGR_FLAVOR "${Pin}"
  echo "COD toolchain hygiene (compiler-critical rows must read [COD]):"
  codToolchainRow clang++ "$(command -v clang++)" 1 || Rc=1
  codToolchainRow comgr "${Comgr}" 1 || Rc=1
  if [[ "${ComgrVer}" != "?" && ! -e "${RocmRoot}/.info/version" ]] && underCod "${Comgr}"; then
    ComgrVersionTrusted=0
  fi
  # rocke_cod_smoke.py pins rocKE's vintage lookup only when it cannot be
  # believed; pinning it otherwise would also satisfy rocKE's IR-flavor guard
  # and hide a genuine comgr-vs-clang split.
  export ROCKE_COMGR_VERSION_TRUSTED="${ComgrVersionTrusted}"
  if (( ComgrVersionTrusted == 1 )); then
    echo "                       comgr interface ${ComgrIface} ($(basename "$(realpath -m "${Comgr}" 2>/dev/null)")), rocm vintage ${ComgrVer} -> rocke flavor ${ComgrFlavor}"
  else
    echo "                       comgr interface ${ComgrIface} ($(basename "$(realpath -m "${Comgr}" 2>/dev/null)")), rocm vintage metadata unavailable in COD"
  fi
  echo "                       cod clang emits the ${ClangShape} p8 datalayout"
  # The probe reports both bases: the flavor the comgr's ROCm number implies and the
  # one it chose from the clang's datalayout generation. They differ only when those
  # two disagree, which means one of them is not from this install -- and the clang
  # wins, because a datalayout cannot leak from an unrelated tree.
  if (( ComgrVersionTrusted == 1 )) && [[ "${ComgrFlavor}" != "?" && "${Pin}" != "${ComgrFlavor}" ]]; then
    echo "WARNING: datalayout split -- comgr rocm ${ComgrVer} implies ${ComgrFlavor}, but the COD"
    echo "         clang emits the ${ClangShape} p8 shape; pinning ${Pin} to match the clang."
    # A row, not only a warning: this says the comgr and the clang in one install
    # disagree about the IR they speak, which is the sharpest packaging signal this
    # gate produces, and the dashboard never sees an echo.
    rockeResult setup cod-datalayout-split 1 \
      "comgr rocm ${ComgrVer} implies ${ComgrFlavor} but the COD clang emits the ${ClangShape} p8 shape; pinned ${Pin}" \
      compiler
  fi
  # A COD shipping no .info/version lets rocke's vintage number leak from the
  # system /opt/rocm. Only worth saying for a COD-resident comgr: an external one
  # already failed hard above.
  if (( ComgrVersionTrusted == 0 )); then
    # Also a row: on a COD that ships no .info/version this is the branch that
    # fires, and it means rocKE keys its feature decisions off a foreign vintage --
    # which is how a compile the COD can do gets refused. Packaging, hence compiler.
    rockeResult setup cod-vintage-leak 1 \
      "COD ships no .info/version, so the comgr vintage ${ComgrVer} came from the system installation; rocKE will gate features on it" \
      compiler
    echo "WARNING: ignoring comgr rocm vintage ${ComgrVer}: it leaked from the system /opt/rocm fallback"
    echo "         ($(cat /opt/rocm/.info/version 2>/dev/null || echo '?')); the flavor knobs keep whatever rocKE derived from it."
  fi
  # A hard requirement is a property of the lanes actually running, so 'all'
  # expands to its lane list and the lane -> tool mapping is stated once.
  local Lane Running="${Stage}"
  [[ "${Stage}" == all ]] && Running="${ROCKE_ALL_LANES}"
  # shellcheck disable=SC2086 # intended word splitting of the lane list
  for Lane in ${Running}; do
    case "${Lane}" in
      pytest|gpu-numeric) HipHard=1; HipccHard=1 ;;
      cod-comgr)          HipHard=1 ;;
      perf)               ReadelfHard=1 ;;
      # rocKE builds the engine archive by invoking `c++`; the shim points that name
      # at the COD, and this row is what proves it rather than assuming it.
      engine)             CxxHard=1 ;;
    esac
  done
  codToolchainRow hip-runtime "${ROCKE_HIP_LIB}" "${HipHard}" || Rc=1
  codToolchainRow hipcc "$(command -v hipcc)" "${HipccHard}" || Rc=1
  codToolchainRow llvm-readelf "$(command -v llvm-readelf)" "${ReadelfHard}" || Rc=1
  codToolchainRow c++ "$(command -v c++)" "${CxxHard}" || Rc=1
  (( Rc == 0 )) || fatalSetup \
    "compiler toolchain resolves outside the COD (${RocmRoot}); refusing to test a stale system ROCm" \
    toolchain
  reportArchDrift
}

# Say when rocKE wires a target this suite does not sweep, or sweeps one it has
# dropped. Which targets to cover is this team's policy, so rocKE's list is not
# adopted wholesale: it carries entries our sweep deliberately omits, such as the
# non-physical gfx11-generic, and taking it would both change the row population and
# hand coverage policy to upstream. Asking it is still how we hear about a target
# arriving, which a hardcoded list can only tell us by staying silent.
#
# Reported as unmeasured rather than red: the difference is a decision waiting to be
# made, not a fault, and a red row that nobody can resolve is the thing this suite
# has spent the most effort removing.
function reportArchDrift {
  local Wired Arch Ours Unswept=""
  Wired="$("${PyBin}" -c 'from rocke.core.isa.backend import wired_arches
print(" ".join(wired_arches()))' 2>/dev/null)" || return 0
  [[ -n "${Wired}" ]] || return 0
  Ours=" ${ROCKE_CI_ARCHES} ${ROCKE_CI_ARCHES_EXPERIMENTAL} "
  for Arch in ${Wired}; do
    [[ "${Ours}" == *" ${Arch} "* ]] || Unswept+="${Unswept:+ }${Arch}"
  done
  [[ -n "${Unswept}" ]] && rockeResult setup arch-coverage Check \
    "rocKE wires ${Unswept}, which this suite does not sweep (ROCKE_CI_ARCHES)" harness
  return 0
}
