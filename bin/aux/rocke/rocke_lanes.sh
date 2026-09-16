# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# The lanes: what each one runs, what it is worth to the compiler team, and
# the floor below which its row count is not credible.
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

# Flavors the byte-identity gate sweeps. 'auto' asks rocKE for its own published
# list, so a flavor it adds is swept the night it appears instead of waiting for
# someone here to notice; the fallback is the pair that predates the list.
#
# Reports through a global for the same reason ensureEngineExtension does: it may
# emit a result row, and a caller capturing stdout would read the row itself as the
# answer -- eight words of a red row swept as eight flavors.
function engineFlavors {  # sets EngineFlavorList
  if [[ "${ROCKE_ENGINE_FLAVORS}" != auto ]]; then
    EngineFlavorList="${ROCKE_ENGINE_FLAVORS}"; return
  fi
  EngineFlavorList="$("${PyBin}" -c 'from rocke.core.lower_llvm import LLVM_FLAVORS
print(" ".join(LLVM_FLAVORS))' 2>/dev/null)"
  if [[ -z "${EngineFlavorList}" ]]; then
    EngineFlavorList="llvm20 llvm22"
    rockeResult setup engine-flavors 1 \
      "cannot read rocKE's LLVM_FLAVORS; sweeping ${EngineFlavorList} only" harness
  fi
}

# How much the byte-identity gate covered, and what it faulted on, so a result row
# can carry both instead of pointing at the log.
# Echoes the count, or nothing when the gate did not report one. Both are red, but
# they are different findings and the row must not blame rocKE's corpus for a change
# in how the gate prints.
function gateFamilyCount {  # <gate-log>
  grep -m1 -oP 'families=\K[0-9]+' "${1}" 2>/dev/null || true
}

function gateFailures {  # <gate-log>
  local Bad
  # Anchored on the per-family line only. The gate also prints "<TAG> <count>" and
  # "<TAG> families: a, b" with the same leading tags, and a looser pattern
  # harvested "2" and "families:" into the row instead of family names.
  Bad="$(grep -oP '^ {2}(DRIFT|RANGE_DRIFT|COMPILE_FAIL|MODE_UNSUPPORTED) {2,}\K[A-Za-z_][A-Za-z0-9_]*' \
         "${1}" 2>/dev/null | sort -u | head -5 | tr '\n' ' ')"
  printf '%s' "${Bad:-no family named by the gate; see the lane log}"
}

function stageEngine {
  requireCmake || return
  # The C++ archive is flavor-independent -- the flavor only changes run_diff's .ll
  # emission -- so every flavor gets the same --build-root and cmake's incrementality
  # covers the rebuilds after the first. The gate reconfigures each time regardless;
  # this only avoids recompiling.
  local -a Flavors
  engineFlavors
  read -ra Flavors <<< "${EngineFlavorList}"
  local Flavor
  for Flavor in "${Flavors[@]}"; do
    echo "byte-identity gate: ${Flavor}"
    local Gate="${BuildRoot}/byte-identity-${Flavor}.log"
    # Keep a copy of the gate's own output so the row can name what it faulted on;
    # PIPESTATUS because tee would otherwise report its own success as the verdict.
    ROCKE_LLVM_FLAVOR="${Flavor}" "${PyBin}" "${ROCKE_TOP}/tools/check_byte_identity.py" \
      --build-root "${BuildRoot}" 2>&1 | tee "${Gate}"
    if (( PIPESTATUS[0] == 0 )); then
      # An empty corpus passes this gate trivially, so the coverage it claims is part
      # of the verdict rather than decoration: too few families is a red row, however
      # green the comparison was.
      local Fams; Fams="$(gateFamilyCount "${Gate}")"
      if [[ -z "${Fams}" ]]; then
        rockeResult byte-identity "${Flavor}" 1 \
          "gate passed but reported no family count: cannot tell what it covered" \
          "${LaneRelevance}"
      elif (( Fams < ROCKE_MIN_GATE_FAMILIES )); then
        rockeResult byte-identity "${Flavor}" 1 \
          "gate passed over only ${Fams} families (floor ${ROCKE_MIN_GATE_FAMILIES}): its corpus shrank" \
          "${LaneRelevance}"
      else
        rockeResult byte-identity "${Flavor}" 0 \
          "engine == python .ll over ${Fams} families" "${LaneRelevance}"
      fi
    else
      # Name the families that drifted: "see log" sends the reader into a
      # thousand-line file for something a one-line message can carry.
      rockeResult byte-identity "${Flavor}" 1 \
        "gate RED: $(gateFailures "${Gate}")" "${LaneRelevance}"
    fi
  done
}

function stageCtest {
  local RunRc
  requireCmake || return
  echo "cmake configure"
  cmake -S "${ROCKE_TOP}" -B "${BuildRoot}" -DCMAKE_BUILD_TYPE=Release \
    || { rockeResult ctest configure 1 "cmake configure failed" "${LaneRelevance}"; return; }
  echo "cmake build"
  cmake --build "${BuildRoot}" -j"$(nproc 2>/dev/null || echo 4)" \
    || { rockeResult ctest build 1 "cmake build failed" "${LaneRelevance}"; return; }
  echo "ctest"
  local Xml="${BuildRoot}/ctest-junit.xml"
  rm -f "${Xml}"  # never let a reused build dir's stale report stand in
  ( cd "${BuildRoot}" && ctest --output-on-failure --no-tests=ignore --output-junit "${Xml}" )
  RunRc=$?
  ExpectedRunnerFailure=8 emitJunit "${Xml}" ctest "${RunRc}"
}

function stagePytest {
  local RunRc Root
  local -a TestRoots=() Missing=()
  # A root pytest cannot collect from is a usage error that ends the lane before
  # any test runs, so the ~1000 rows vanish instead of turning red. Report the
  # gap and run the roots that are usable.
  # Every root is reported the same way when it disappears. Treating one as optional
  # is how a whole tree leaves the suite unnoticed -- and library/tests is precisely
  # the tree rocKE's own gap registry says nothing else gates.
  for Root in "${ROCKE_TOP}/tests" \
              "${ROCKE_TOP}/python/rocke/benchmark" \
              "${ROCKE_TOP}/python/rocke/heuristics/tests" \
              ; do
    if hasTests "${Root}"; then TestRoots+=("${Root}"); else Missing+=("${Root}"); fi
  done
  # library/tests belongs to the surrounding rocm-libraries checkout, so it is only
  # required when we are in one: ROCKE_TOP is documented as accepting a bare rocKE
  # tree, and demanding it there would make that mode permanently red.
  #
  # It stays in the same pytest invocation as the roots above. Running it in its own
  # process to isolate torch from rocKE's HIP runtime looked attractive, but pytest
  # derives a nodeid from the argument set: dropping one root rewrote 1226 of 2704
  # classnames, so every one of those rows would leave the dashboard and return under
  # a new name. The ordering is fixed in-process instead, by rocke_relevance.py's
  # claim_device_for_torch().
  if [[ "${ROCKE_REPO_ROOT}" != "${ROCKE_TOP}" ]]; then
    Root="${ROCKE_PROJECT_ROOT}/library/tests"
    if hasTests "${Root}"; then TestRoots+=("${Root}"); else Missing+=("${Root}"); fi
  fi
  if (( ${#Missing[@]} )); then
    rockeResult setup pytest-roots 1 "unusable test roots: ${Missing[*]}"
  fi
  if (( ${#TestRoots[@]} == 0 )); then
    rockeResult setup pytest-roots 1 "no pytest roots under ${ROCKE_TOP}"
    return
  fi
  ensureProjectTestDeps || return
  # rocKE's cross-engine tests need its C++ extension; without it they skip, and a
  # skip naming the toolchain is a blocked row. Its absence is reported by the
  # builder itself, so a failure here degrades the lane instead of ending it.
  ensureEngineExtension || true
  echo "relative-path guard"
  if "${PyBin}" "${ROCKE_TOP}/tests/run_all.py" --no-gate --no-pytest \
      --build-root "${BuildRoot}/guard"; then
    rockeResult guard relative-path 0 ok "${LaneRelevance}"
  else
    rockeResult guard relative-path 1 "guard failed" "${LaneRelevance}"
  fi
  echo "pytest (project unit-test roots)"
  local Xml="${BuildRoot}/pytest-junit.xml"
  local Manifest="${BuildRoot}/pytest-relevance.json"
  local Out="${BuildRoot}/pytest-output.log"
  runPytest "${Xml}" "${Manifest}" "${Out}" "${TestRoots[@]}" \
    --ignore="${ROCKE_TOP}/tests/instances/test_rocke_numeric.py"
  RunRc=$?
  emitJunit "${Xml}" pytest "${RunRc}" "${Manifest}" "${Out}"
}

function stageGpuNumeric {
  local DeviceArch RunRc
  # Pin the device here rather than in a wrapper: this lane launches kernels, and
  # the 'all' wrapper runs it too, so a wrapper-only export made the consolidated
  # nightly and the standalone lane measure different GPUs on a multi-GPU host.
  # The probe below asks about index 0 of whatever is visible, so both agree.
  export ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-0}"
  # "This host has no GPU" is a green skip; "we could not ask" is not. Collapsing
  # the two would let a rocKE rename retire the only lane that can catch a
  # miscompile without a single red row, so the probe reports which happened.
  # stdout only: the probe already reports its own failure there, and folding
  # stderr in would let one HSA warning line prepend itself to the value the arms
  # below match -- promoting noise to "this is the device".
  DeviceArch="$("${PyBin}" - <<'PY'
try:
    from rocke.runtime.hip_module import get_device_arch

    print(get_device_arch(0) or "none")
except Exception as exc:  # noqa: BLE001 - any failure here means "cannot ask"
    print(f"error: {type(exc).__name__}: {exc}".replace("\n", "; "))
PY
)"
  case "${DeviceArch}" in
    none)
      rockeResult environment device Check "no ROCm GPU agent on this host"
      return ;;
    gfx[0-9a-f]*) ;;
    *)
      # Anything that is not "none" and not an arch is a broken probe, including
      # the empty string and any unexpected chatter.
      rockeResult environment device 1 \
        "cannot query the device through rocKE: ${DeviceArch:-no output}" "${LaneRelevance}"
      return ;;
  esac
  # Past the device check this host *can* certify numerics, and this is the only
  # lane that can catch a miscompile, so a missing reference must never read as
  # coverage. It is still worth telling two cases apart, because one is a defect
  # and the other is a host nobody prepared:
  #
  #   present but unusable -- a non-ROCm build, or one from the wrong datalayout
  #     generation -- is red wherever it happens. Someone prepared this host, and
  #     prepared it wrongly.
  #   absent is red on the host that is supposed to certify numerics
  #     (ROCKE_NUMERIC_HOST=1) and unmeasured anywhere else. Reddening every host
  #     that was never meant to run numerics produced a nightly red nobody could
  #     act on, which is how a tier stops being read.
  #
  # Either way the row names the one command that fixes it.
  if ! ensureTorch; then
    local Fix Replace
    Fix="install a ROCm torch in ${ROCKE_VENV} (same datalayout generation as the COD, e.g. $(rocmWheelIndex 7.2)) or set ROCKE_TORCH_INDEX_URL"
    # Replacing, not installing: pip counts an installed-but-wrong torch as
    # satisfying the requirement, so an index alone will not repair this one.
    Replace="uninstall it and install a ROCm torch of the COD's datalayout generation in ${ROCKE_VENV}, e.g. from $(rocmWheelIndex 7.2)"
    # Installed is not the same question as importable: a torch whose shared
    # libraries are missing fails to import, and reading that as "no torch here"
    # would downgrade a broken installation to an unprepared host.
    if "${PyBin}" -c 'import importlib.util,sys; sys.exit(0 if importlib.util.find_spec("torch") else 1)' 2>/dev/null; then
      rockeResult environment torch 1 \
        "the torch in ${ROCKE_VENV} cannot serve as a numeric reference for this COD: ${Replace}"
    elif [[ "${ROCKE_NUMERIC_HOST}" == 1 ]]; then
      rockeResult environment torch 1 \
        "no numeric reference on a ${DeviceArch} host declared ROCKE_NUMERIC_HOST=1: ${Fix}"
    else
      rockeResult environment torch Check \
        "no numeric reference on this ${DeviceArch} host, so nothing was certified: ${Fix}"
    fi
    return
  fi
  # Every case here emits a kernel, compiles it through the COD and launches it on
  # the device -- rocKE runs each in a child process, so the in-process probe sees
  # none of it and would otherwise file the most compiler-driven rows we have as
  # 'logic'.
  # Read by emitJunit in rocke_env.sh through dynamic scope, which is why it is a
  # local here and looks unused in this file. The lane guarantees every case it runs
  # drives the toolchain, and the floor is how a row with no measured evidence still
  # says so rather than reading as unrelated logic.
  # shellcheck disable=SC2034
  local LaneRelevanceFloor=compiler
  echo "numeric certification: ${DeviceArch}"
  local Xml="${BuildRoot}/numeric-junit.xml"
  local Manifest="${BuildRoot}/numeric-relevance.json"
  local Out="${BuildRoot}/numeric-output.log"
  runPytest "${Xml}" "${Manifest}" "${Out}" \
    "${ROCKE_TOP}/tests/instances/test_rocke_numeric.py"
  RunRc=$?
  emitJunit "${Xml}" "numeric-${DeviceArch}" "${RunRc}" "${Manifest}" "${Out}"
}

function stageCodOccupancy {
  # Host-only codegen signal: per arch, compile the smoke kernel with the COD
  # comgr and read its resource footprint from the HSACO's ELF notes -- no GPU,
  # no torch. A spill on this fixed kernel is a real regression. See README.
  echo "codegen resource footprint (native rocke.benchmark.perf.occupancy)"
  codSmokeSweep occupancy
}

# Run the single-arch cod smoke for one arch. Extra args (e.g. --experimental)
# are forwarded to the driver; the aborted-row keeps the same experimental tag.
function codSmoke {  # <mode> <arch> [extra driver args...]
  local Mode="${1}" Arch="${2}"; shift 2
  local -a Args=(--mode "${Mode}" --arch "${Arch}" --flavor)
  if [[ "${Mode}" == codegen ]]; then
    Args+=("${ROCKE_CODEGEN_FLAVOR}" --clang "${AOMP}/bin/clang" --out "${BuildRoot}")
  else
    Args+=("${ROCKE_COMGR_FLAVOR}")
    [[ "${Mode}" == occupancy ]] \
      && Args+=(--readelf "${AOMP}/bin/llvm-readelf")
  fi
  # Mirror the driver's grouping so a hard abort files where its own rows would;
  # a plain universal_gemm would collide across modes. See rocke_cod_smoke.py.
  local Group="universal_gemm.${Mode}"
  [[ "${Mode}" == occupancy ]] && Group="occupancy"
  local Suffix=""; [[ "$*" == *--experimental* ]] && Suffix=" (experimental)"
  "${PyBin}" "${HelperDir}/rocke_cod_smoke.py" "${Args[@]}" "$@" \
    || rockeResult "${Group}" "${Arch}${Suffix}" 1 "${Mode} driver aborted (see log)" \
         "${LaneRelevance}"
}

# Sweep the production arches, then the experimental ones (tagged as such).
function codSmokeSweep {  # <mode>
  local Mode="${1}"
  local -a Prod Experimental
  local Arch
  read -ra Prod <<< "${ROCKE_CI_ARCHES}"
  read -ra Experimental <<< "${ROCKE_CI_ARCHES_EXPERIMENTAL}"
  for Arch in "${Prod[@]}"; do codSmoke "${Mode}" "${Arch}"; done
  for Arch in "${Experimental[@]}"; do codSmoke "${Mode}" "${Arch}" --experimental; done
}

function stageCodCodegen { codSmokeSweep codegen; }

function stageCodComgr { codSmokeSweep comgr; }

# Origin of a report row: 'rocKE' = the project's own tests/tools, 'ci-harness'
# = a probe this CI adds (cod-*) or its own plumbing, whichever lane hit it.
# Mirrors rocke_extract.py's area classifier. See README.md "Test origin".
# Everything the run needs to know about a lane, in run order, one line each:
#
#   name | origin | relevance | floor | tools that must be the COD's
#
# One table because the same seven lanes used to be spelled out in six places --
# an order list, a dispatch, an origin map, a relevance map, a floor table and a
# tool-hardness map -- and a lane added to five of them looked like it worked.
#
# The handler is not a field: it is derived from the name by laneHandler, so the
# table cannot name the wrong one. It could, once. Pointing the occupancy lane's
# handler at stageCodCodegen passed the "is it a function?" check, ran codegen
# twice, emitted no occupancy row at all, and still cleared its floor, because
# both lanes size
# their floor by the arch sweep -- a whole check gone, with nothing red to show it.
#
# floor is the row count below which the lane is not credible: a number, or a token
# for the counts only known at run time (flavors, arches). Tools is a comma list,
# empty when the lane needs nothing beyond clang and comgr, which every lane needs.
LaneRegistry=(
  "engine|rocKE|compiler|flavors|c++"
  "ctest|rocKE|compiler|3|"
  "pytest|rocKE|compiler-capable|400|hip-runtime,hipcc"
  "cod-codegen|ci-harness|compiler|arches|"
  "cod-comgr|ci-harness|compiler|arches|hip-runtime"
  "gpu-numeric|rocKE|compiler-capable|5|hip-runtime,hipcc"
  "cod-occupancy|ci-harness|compiler|arches|llvm-readelf"
)

# Field values the table may carry. Named here so a typo is caught at startup
# instead of turning into a silently mis-filed row: an unknown origin would
# misattribute a failure between rocKE and this CI, and an unknown relevance sorts
# last and drops out of the compiler count. Floors validate themselves: laneRowFloor
# answers "?" for anything that is neither a token nor a count. The relevance list
# mirrors rocke_tiers.py, which bash cannot import.
LaneOrigins="rocKE ci-harness"
LaneRelevances="compiler compiler-capable logic harness unmeasured"
LaneKnownTools="c++ hip-runtime hipcc llvm-readelf"

# The stage function for a lane: cod-codegen -> stageCodCodegen. Derived rather
# than stored, so a lane can only ever run its own body.
function laneHandler {  # <lane>
  local Part Name=""
  for Part in ${1//-/ }; do Name+="${Part^}"; done
  printf 'stage%s\n' "${Name}"
}

# Field <n> of <lane>'s row, empty when the lane is not registered. Callers that
# must not accept an unregistered lane check for empty rather than defaulting,
# because a plausible default is what let a half-added lane pass unnoticed.
function laneField {  # <lane> <1-based field>
  local Row
  for Row in "${LaneRegistry[@]}"; do
    [[ "${Row%%|*}" == "${1}" ]] || continue
    cut -d'|' -f"${2}" <<< "${Row}"
    return 0
  done
  return 1
}

function laneNames {
  local Row
  for Row in "${LaneRegistry[@]}"; do printf '%s\n' "${Row%%|*}"; done
}

function laneOrigin {  # <lane> [group]
  case "${2:-}" in setup|environment) echo ci-harness; return ;; esac
  laneField "${1}" 2 || echo ci-harness
}

# How a red row from a lane should be triaged when there is no per-test evidence
# for it. The COD lanes drive the compiler by construction, so they are
# 'compiler' outright; the pytest lanes measure it per test (rocke_relevance.py)
# and fall back to 'compiler-capable' -- never 'logic' -- so an unmeasured row is
# always looked at rather than silently written off. See README.md
# "Test relevance".
function laneRelevance {  # <lane>
  laneField "${1}" 3 || echo unregistered
}

# The tools this lane must find inside the COD, beyond clang and comgr which every
# lane needs. Hardness follows the lanes actually running, so a host missing hipcc
# still runs the lanes that never call it.
function laneHardTools {  # <lane>
  laneField "${1}" 5 | tr ',' ' '
}

# Rows below which a lane cannot be healthy, and a red row when it falls there.
#
# Everything else in this driver reports what it *did*; nothing noticed a lane that
# quietly stopped doing anything. rocKE relocates test trees routinely, and a root
# that keeps one file behind takes the pytest lane from ~1150 rows to 1 -- all green,
# because every row that ran passed. The house reads silence as success, so shrinking
# coverage has to be an explicit failure.
#
# These are floors, not expectations: set far below today's counts so ordinary
# upstream churn never trips them, and only a collapse does. The arch and flavor
# lanes are derived from the lists they iterate, so they need no maintenance; the two
# absolute numbers are the price of noticing a corpus disappear, and a legitimate
# shrink below them is a one-line edit here with the reason in the commit.
# Resolve the registry's floor field. A '?' is not a benign default: it means the
# lane reached here without a row in the table, and a floor of 1 would have hidden
# that behind a green row.
function laneRowFloor {  # <lane>
  local Floor; Floor="$(laneField "${1}" 4)" || { echo "?"; return; }
  local -a Arches Experimental Flavors
  case "${Floor}" in
    flavors)
      read -ra Flavors <<< "${EngineFlavorList:-${ROCKE_ENGINE_FLAVORS}}"
      echo "${#Flavors[@]}" ;;
    # Both sweeps, because codSmokeSweep runs the experimental arches too: counting
    # only the production ones let every experimental row vanish inside the floor.
    arches)
      read -ra Arches <<< "${ROCKE_CI_ARCHES}"
      read -ra Experimental <<< "${ROCKE_CI_ARCHES_EXPERIMENTAL}"
      echo "$(( ${#Arches[@]} + ${#Experimental[@]} ))" ;;
    # Anything that is neither a token nor a plain count, including the empty
    # string a missing field yields, is not a floor.
    ''|*[!0-9]*) echo "?" ;;
    *)           echo "${Floor}" ;;
  esac
}

function assertRowFloor {  # <lane> <row-log> <relevance>
  local Lane="${1}" Log="${2}" Relevance="${3}" Floor Rows
  Floor="$(laneRowFloor "${Lane}")"
  if [[ "${Floor}" == "?" ]]; then
    rockeResult setup "${Lane}-coverage" 1 \
      "lane ${Lane} has no row floor: give it one in LaneRegistry" harness
    return
  fi
  # A lane that reported an environment blocker (no GPU, no numeric reference) never
  # reached its tests, so a low row count is that row's news and not a second finding.
  grep -aq '^ROCKE_RESULT|environment|' "${Log}" 2>/dev/null && return 0
  Rows="$(grep -ac '^ROCKE_RESULT|' "${Log}" 2>/dev/null || true)"
  (( ${Rows:-0} < Floor )) || return 0
  rockeResult setup "${Lane}-coverage" 1 \
    "only ${Rows:-0} result rows, below this lane's floor of ${Floor}: its corpus shrank" \
    "${Relevance}"
}

# Each lookup used to fall through to a plausible-looking default: a lane missing
# from the relevance map was triaged as our own plumbing, one missing from the floor
# table was floored at a single row. Both are silent, which is the one failure mode
# this driver is not allowed to have, so a gap is a startup error.
# Prove the registry says something valid, not merely something. A field that is
# absent, duplicated or misspelt has to stop the run here: every one of those turns
# into a row that looks like a result later, and the point of this suite is that a
# row means what it says.
#
# Checked before anything runs, including before 'all' forks its children, so a
# malformed table fails once at startup rather than seven times inside lanes that
# have already claimed to test something.
function assertLaneTables {
  local Row Lane Tool Seen=" " Bad=""
  for Row in "${LaneRegistry[@]}"; do
    Lane="${Row%%|*}"
    # cut reports success and an empty string for a field that is not there, so
    # the row's shape is counted rather than inferred from a lookup.
    (( $(awk -F'|' '{print NF}' <<< "${Row}") == 5 )) || { Bad+=" fields:${Lane:-<empty>}"; continue; }
    [[ -n "${Lane}" ]] || { Bad+=" name:<empty>"; continue; }
    [[ "${Seen}" != *" ${Lane} "* ]] || Bad+=" duplicate:${Lane}"
    Seen+="${Lane} "
    [[ " ${LaneOrigins} " == *" $(laneField "${Lane}" 2) "* ]] || Bad+=" origin:${Lane}"
    [[ " ${LaneRelevances} " == *" $(laneField "${Lane}" 3) "* ]] || Bad+=" relevance:${Lane}"
    [[ "$(laneRowFloor "${Lane}")" != "?" ]] || Bad+=" floor:${Lane}"
    for Tool in $(laneHardTools "${Lane}"); do
      [[ " ${LaneKnownTools} " == *" ${Tool} "* ]] || Bad+=" tool:${Lane}:${Tool}"
    done
    declare -F "$(laneHandler "${Lane}")" >/dev/null || Bad+=" handler:${Lane}"
  done
  [[ -z "${Bad}" ]] \
    || fatalSetup "the lane registry is not usable:${Bad}" harness
}
