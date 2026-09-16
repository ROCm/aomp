# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Resolving the compiler under test, and proving every tool comes from it.
#
# Sourced by ../../run_rocke.sh, which owns the run. Sourcing defines functions and
# constant tables; it starts nothing, touches no file and cannot fail a run.
# shellcheck shell=bash
#
# The driver owns the run's state -- RocmRoot, PyBin, BuildRoot, Stage,
# LaneRelevance and the rest -- which these functions read, and in a few cases
# set for the driver to use later (setupPython assigns PyBin, engineFlavors
# assigns EngineFlavorList, the toolchain functions export the flavor knobs).
# Checked on its own, shellcheck cannot see where that state comes from, so
# SC2154 is off for the file; check the driver too, since -x follows a source
# for definitions but reports nothing inside it.
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

# Populate the shim the driver already named in PATH. Separate from that line
# because filling it needs fatalSetup, which cannot run until the driver has set up
# its row channel: PATH is built while the environment is still being decided.
function installCodShim {
  if ! mkdir -p "${CodShim}" \
    || ! ln -sf "${AOMP}/bin/clang++" "${CodShim}/c++" \
    || ! ln -sf "${AOMP}/bin/clang" "${CodShim}/cc"; then
    fatalSetup "cannot create the COD compiler shim in ${CodShim}" harness
  fi
}

# The HIP runtime the COD ships, which decides half of what an on-device row means.
function codHipVersion {
  awk -F= '
    /^HIP_VERSION_(MAJOR|MINOR|PATCH|GITHASH)=/ { v[$1] = $2 }
    END { if (v["HIP_VERSION_MAJOR"] != "")
            printf "%s.%s.%s-%s", v["HIP_VERSION_MAJOR"], v["HIP_VERSION_MINOR"], \
                                  v["HIP_VERSION_PATCH"], v["HIP_VERSION_GITHASH"] }
  ' "${RocmRoot}/share/hip/version" 2>/dev/null
}

# Said before anything is refreshed or built, so a run that dies early still names
# what it was testing. The rest of the provenance waits for printProvenance, where
# the gate has resolved the parts that are still 'auto' here.
function printBanner {
  echo "==============================================================================="
  echo "rocKE ${Stage}  ($(date '+%Y-%m-%d %H:%M:%S'))"
  echo "  AOMP        = ${AompInput} -> ${AOMP}"
  echo "  ROCM_PATH   = ${ROCM_PATH} (${RocmRootSource})"
  echo "==============================================================================="
}

# True when a resolved path lives inside the COD install root.
# Everything that decides what a row means, in one block, once the gate has resolved
# it. Scattered provenance is provenance nobody reads: the flavor knobs still said
# 'auto' in the opening banner because they are pinned later, the comgr facts were
# echoed mid-gate, and which torch produced a GPU result was not recorded anywhere.
#
# The rule this serves is the suite's oldest: a green row must name what produced it.
# That means the compiler, but also the packages a verdict depends on -- pytest
# decides what ran and how it is counted, pytest-subtests decides the shape of the
# report this CI reconciles against, numpy and torch are the numeric references, and
# pybind11 builds the engine extension the cross-engine tests compare. A version
# change in any of them can move a row without the compiler moving at all.
function printProvenance {
  local Dev Driver Os
  echo "== provenance ================================================================="
  # Sourced in a subshell: /etc/os-release sets a dozen common names, and this
  # function must not take NAME or VERSION from the host into the run's scope.
  Os="$( . /etc/os-release 2>/dev/null && printf '%s' "${PRETTY_NAME:-${NAME:-} ${VERSION_ID:-}}" )"
  echo "  host        $(uname -n)  ${Os:-$(uname -s)}, kernel $(uname -r)"
  Driver="$(cat /sys/module/amdgpu/version 2>/dev/null)"
  # The node is often shared, so which device this run was given is part of the
  # result: two runs on one host can see different GPUs.
  Dev="$("${PyBin}" - <<'PY' 2>/dev/null
try:
    from rocke.runtime.hip_module import get_device_arch
    print(get_device_arch(0) or "no GPU agent")
except Exception as exc:  # noqa: BLE001
    print(f"cannot ask rocKE: {type(exc).__name__}")
PY
)"
  echo "  device      ${Dev:-?}  (ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES:-<unset>}${Driver:+, amdgpu ${Driver}})"
  echo "  COD         ${RocmRoot}"
  echo "              clang $("${CXX}" --version 2>/dev/null | head -1 | sed 's/^AMD clang version //;s/ (http.*//')" \
       "(llvm $("${CXX}" --version 2>/dev/null | grep -oE '[0-9a-f]{12,40}' | tail -1 | cut -c1-12))"
  echo "              HIP $(codHipVersion)"
  echo "              comgr ${CodComgrIface:-?} ($(basename "$(realpath -m "${CodComgrLib:-?}" 2>/dev/null)"))," \
       "rocm $(if (( ${ComgrVersionTrusted:-1} == 1 )); then echo "${CodComgrVintage:-?} -> ${CodComgrFlavor:-?}"; else echo "vintage not in the COD"; fi)"
  echo "              emits the ${CodClangShape:-?} p8 datalayout;" \
       "flavors codegen:${ROCKE_CODEGEN_FLAVOR} comgr:${ROCKE_COMGR_FLAVOR}" \
       "engine:$(if [[ "${ROCKE_ENGINE_FLAVORS}" == auto ]]; then echo "auto (rocKE's list, swept by the engine lane)"; else echo "${ROCKE_ENGINE_FLAVORS}"; fi)"
  echo "  rocKE       $(rockeSrcRev)  ${ROCKE_TOP}"
  echo "  python      $("${PyBin}" -V 2>&1 | sed 's/^Python //')  ${PyBin}"
  # Reported, never required: a package this suite does not need on every host must
  # not turn its absence into a failure here. The lane that needs one says so itself.
  "${PyBin}" - <<'PY' || true
import importlib.metadata as md
import pathlib
for name in ("pytest", "pytest-subtests", "numpy", "pybind11", "torch"):
    try:
        ver = md.version(name)
    except Exception:  # noqa: BLE001
        print(f"              {name:16} absent")
        continue
    where = ""
    try:
        mod = __import__(name.replace("-", "_"))
        where = str(pathlib.Path(getattr(mod, "__file__", "") or "").parent)
    except Exception:  # noqa: BLE001
        where = "(not importable)"
    hip = ""
    if name == "torch":
        try:
            import torch
            hip = f" hip {torch.version.hip}" if torch.version.hip else " not a ROCm build"
        except Exception:  # noqa: BLE001
            hip = " (not importable)"
    print(f"              {name:16} {ver}{hip}  {where}")
PY
  echo "  build       cmake $(cmake --version 2>/dev/null | head -1 | awk '{print $3}')," \
       "ninja $(ninja --version 2>/dev/null || echo absent)"
  echo "==============================================================================="
}

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
  local Probe Pin Rc=0
  local HipHard=0 HipccHard=0 ReadelfHard=0 CxxHard=0 ComgrVersionTrusted=1
  # rocke_cod_probe.py reports the comgr lib rocKE will actually load, its ROCm
  # vintage, the IR flavor *rocKE derives* from that vintage, and the lib's own
  # interface version. The flavor comes from rocKE's own ladder, so a release that
  # adds a flavor needs no edit here.
  CodClangShape="$(codClangP8Shape "${ROCKE_CI_ARCHES%% *}")"
  # The clang datalayout is the one basis for the flavor that cannot leak in from an
  # unrelated tree. Losing it means the flavor falls back to the comgr's ROCm number,
  # which is exactly the value this gate spends thirty lines distrusting -- so say so
  # in a row instead of continuing on the weaker basis in silence. A new p8 shape
  # upstream lands here, and that is worth a night's attention.
  if [[ "${CodClangShape}" == unknown ]]; then
    rockeResult setup cod-datalayout 1 \
      "the COD clang emits a p8 datalayout rocKE does not describe: pinning the flavor from the comgr's ROCm number instead" \
      harness
  fi
  Probe="$("${PyBin}" "${HelperDir}/rocke_cod_probe.py" "${CodClangShape}" 2>/dev/null)"
  read -r Pin CodComgrFlavor CodComgrVintage CodComgrIface CodComgrLib <<< "${Probe}"
  resolveFlavorKnob ROCKE_CODEGEN_FLAVOR "${Pin}"
  resolveFlavorKnob ROCKE_COMGR_FLAVOR "${Pin}"
  echo "COD toolchain hygiene (compiler-critical rows must read [COD]):"
  codToolchainRow clang++ "$(command -v clang++)" 1 || Rc=1
  codToolchainRow comgr "${CodComgrLib}" 1 || Rc=1
  if [[ "${CodComgrVintage}" != "?" && ! -e "${RocmRoot}/.info/version" ]] && underCod "${CodComgrLib}"; then
    ComgrVersionTrusted=0
  fi
  # rocke_cod_smoke.py pins rocKE's vintage lookup only when it cannot be
  # believed; pinning it otherwise would also satisfy rocKE's IR-flavor guard
  # and hide a genuine comgr-vs-clang split.
  export ROCKE_COMGR_VERSION_TRUSTED="${ComgrVersionTrusted}"
  # The probe reports both bases: the flavor the comgr's ROCm number implies and the
  # one it chose from the clang's datalayout generation. They differ only when those
  # two disagree, which means one of them is not from this install -- and the clang
  # wins, because a datalayout cannot leak from an unrelated tree.
  if (( ComgrVersionTrusted == 1 )) && [[ "${CodComgrFlavor}" != "?" && "${Pin}" != "${CodComgrFlavor}" ]]; then
    echo "WARNING: datalayout split -- comgr rocm ${CodComgrVintage} implies ${CodComgrFlavor}, but the COD"
    echo "         clang emits the ${CodClangShape} p8 shape; pinning ${Pin} to match the clang."
    # A row, not only a warning: this says the comgr and the clang in one install
    # disagree about the IR they speak, which is the sharpest packaging signal this
    # gate produces, and the dashboard never sees an echo.
    rockeResult setup cod-datalayout-split 1 \
      "comgr rocm ${CodComgrVintage} implies ${CodComgrFlavor} but the COD clang emits the ${CodClangShape} p8 shape; pinned ${Pin}" \
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
      "COD ships no .info/version, so the comgr vintage ${CodComgrVintage} came from the system installation; rocKE will gate features on it" \
      compiler
    echo "WARNING: ignoring comgr rocm vintage ${CodComgrVintage}: it leaked from the system /opt/rocm fallback"
    echo "         ($(cat /opt/rocm/.info/version 2>/dev/null || echo '?')); the flavor knobs keep whatever rocKE derived from it."
  fi
  # A hard requirement is a property of the lanes actually running, so 'all'
  # expands to its lane list and each lane names its own tools in LaneRegistry.
  # `c++` is there because rocKE builds the engine archive by invoking that bare
  # name; the shim points it at the COD and this row is what proves it.
  local Lane Tool Running="${Stage}"
  [[ "${Stage}" == all ]] && Running="${ROCKE_ALL_LANES}"
  # shellcheck disable=SC2086 # intended word splitting of the lane list
  for Lane in ${Running}; do
    for Tool in $(laneHardTools "${Lane}"); do
      case "${Tool}" in
        hip-runtime)  HipHard=1 ;;
        hipcc)        HipccHard=1 ;;
        llvm-readelf) ReadelfHard=1 ;;
        c++)          CxxHard=1 ;;
        # Not a warning: the lane declared a tool this gate cannot prove belongs
        # to the COD, so every row it goes on to emit would claim a provenance
        # nobody checked. assertLaneTables rejects unknown tokens at startup, so
        # reaching here means the two lists drifted apart.
        *) fatalSetup \
             "lane ${Lane} requires '${Tool}', which the hygiene gate cannot verify" \
             harness ;;
      esac
    done
  done
  codToolchainRow hip-runtime "${ROCKE_HIP_LIB}" "${HipHard}" || Rc=1
  codToolchainRow hipcc "$(command -v hipcc)" "${HipccHard}" || Rc=1
  codToolchainRow llvm-readelf "$(command -v llvm-readelf)" "${ReadelfHard}" || Rc=1
  codToolchainRow c++ "$(command -v c++)" "${CxxHard}" || Rc=1
  (( Rc == 0 )) || fatalSetup \
    "compiler toolchain resolves outside the COD (${RocmRoot}); refusing to test a stale system ROCm" \
    toolchain
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
