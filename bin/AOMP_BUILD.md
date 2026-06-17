# `aomp_build.py` — AOMP build orchestrator

`aomp_build.py` is a unified, introspectable driver for building AOMP. It pulls
the individual taskified component build scripts (`build_<name>.sh`) into a
single workflow: it resolves which components to build from a config file,
orders them by their dependencies, breaks each component into fine-grained
tasks, and runs those tasks (all of them, a numbered range, a glob, or from a
chosen point onward) with per-task logging.

It is the successor workflow to running `build_aomp.sh` directly: the same
components, the same order, but introspectable, incremental, and scriptable.

- Orchestrator: [`bin/aomp_build.py`](aomp_build.py)
- Default config: [`bin/configs/aomp.cudf`](configs/aomp.cudf)
- Per-component scripts: `bin/build_<name>.sh` and `bin/rocmlibs/build_<name>.sh`

---

## Table of contents

- [Requirements](#requirements)
- [Quick start](#quick-start)
- [Concepts](#concepts)
- [Command-line reference](#command-line-reference)
- [Selectors](#selectors)
- [Build variants](#build-variants)
- [Components and features](#components-and-features)
- [The version manifest](#the-version-manifest)
- [Logs](#logs)
- [User workflows](#user-workflows)
- [Internals](#internals)
  - [The taskified component interface](#the-taskified-component-interface)
  - [The CUDF config format](#the-cudf-config-format)
  - [Resolution pipeline](#resolution-pipeline)
  - [Task elaboration](#task-elaboration)
  - [Execution](#execution)
  - [Environment discovery and child environment](#environment-discovery-and-child-environment)
- [Troubleshooting](#troubleshooting)
- [Extending](#extending)

---

## Requirements

- Python 3 (standard library only; no third-party packages).
- `bash` and the usual AOMP build prerequisites (the orchestrator simply
  invokes the existing `build_<name>.sh` scripts).
- A working AOMP environment as set up by `aomp_common_vars` (repos cloned
  under `$AOMP_REPOS`, etc.). `aomp_build.py` itself only reads a few
  variables (`BUILD_DIR`, `AOMP_REPOS`, `AOMP_REPO_NAME`); each build script
  sources `aomp_common_vars` on its own.

`aomp_build.py` does not need to be run from any particular directory — it
locates the build scripts relative to its own location in `bin/`.

---

## Quick start

```bash
cd $AOMP_REPOS/aomp/bin

# See the components that would be built (resolved + dependency-ordered).
./aomp_build.py --components

# See every task that would run, numbered.
./aomp_build.py list

# Dry-run everything (prints commands + log paths, runs nothing).
./aomp_build.py -n

# Build everything.
./aomp_build.py

# Build just one component's tasks.
./aomp_build.py 'comgr/*'

# Resume from a component after fixing a failure (trailing 'continue').
./aomp_build.py comgr/default/cmake continue
```

---

## Concepts

| Term | Meaning |
|------|---------|
| **Component** | A buildable unit with a `build_<name>.sh` script (e.g. `project`, `comgr`, `flang`, `rocBLAS`). |
| **Feature** | A named alias expanding to a set of components (e.g. `flang`, `rocmlibs`), used by `--add`/`--remove`. |
| **Config / variant** | A build configuration a component advertises (e.g. `default`, `asan`, `perf`, `debug`, `*-devicertl`). |
| **Task** | A single step of a component build: `precheck`, `patch`, `clean`, `cmake`, `build`, `install`, `postinstall`, `unpatch`. |
| **Request** | The default set of components to build, declared in the config's `request:` stanza. |
| **Selector** | A positional argument that picks which elaborated tasks to run. |

The pipeline, end to end:

```
parse argv ─▶ load CUDF config ─▶ expand features; apply --add/--remove
           ─▶ dependency closure + topological sort
           ─▶ elaborate tasks (query each build_<name>.sh list, filter by variant)
           ─▶ select tasks (selector grammar)
           ─▶ list, or run with per-task logs
```

---

## Command-line reference

```
aomp_build.py [options] [selector ...]
```

### Actions / output

| Option | Description |
|--------|-------------|
| `list` (selector) | Print the numbered task list and exit. |
| `--components` | Print the resolved, dependency-ordered component list and exit. |
| `-n`, `--dry-run` | Show what would run (command + log path per task) without executing. |
| `--export-manifest [FILE]` | Write a git fingerprint manifest and exit. Default path: `<BUILD_DIR>/manifests/<config>-manifest.json`. |
| `--import-manifest FILE` | Check out recorded git SHAs before building (refuses if any repo is dirty). |

### Component selection

| Option | Description |
|--------|-------------|
| `-c`, `--config FILE` | CUDF config file. Default: `bin/configs/aomp.cudf`. |
| `--add NAMES` | Add component(s) or feature(s). Comma-separated and/or repeatable. |
| `--remove NAMES` | Remove component(s) or feature(s) (cascades to dependents). Comma-separated and/or repeatable. |
| `--variant SPEC` | Variant filter: `cfg` (global) or `comp=cfg` (per-component). Comma-separated and/or repeatable. `default` is always built when offered (so `--variant debug` = default+debug); components offering neither are skipped (so `--variant default` skips the runtimes). See [Build variants](#build-variants). |
| `-C`, `--clean` | Include `clean` tasks (skipped by default for incremental builds). |

### Build environment knobs (exported to child build scripts)

| Option | Environment variable | Notes |
|--------|----------------------|-------|
| `-j`, `--jobs N` | `AOMP_JOB_THREADS` | Parallel build threads. |
| `--ninja` / `--no-ninja` | `AOMP_USE_NINJA=1` / `0` | Use the Ninja generator. |
| `--ccache` / `--no-ccache` | `AOMP_USE_CCACHE=1` / `0` | Use ccache. |
| `--gfx LIST` | `GFXLIST` | GPU target list. |
| `--build-type TYPE` | `BUILD_TYPE` | CMake build type. |
| `--sudo` | `SUDO=yes` | Install with sudo. |

### Logging

| Option | Description |
|--------|-------------|
| `--log-dir DIR` | Directory for per-task logs. Default: `<BUILD_DIR>/aomp_build_logs`. |

Any variable not set via a flag is inherited from the environment, so you can
still drive the build the "classic" way (`AOMP_JOB_THREADS=32 ./aomp_build.py`).

---

## Selectors

Selectors are positional arguments that decide which of the elaborated tasks to
run. The grammar mirrors `amd-build`:

| Selector | Meaning |
|----------|---------|
| *(none)* | Run all elaborated tasks. |
| `list` | Print the numbered task list and exit (does not run anything). |
| `N` | Run task number `N` (1-based, as shown by `list`). |
| `N--M` | Run the inclusive range of tasks `N` through `M`. |
| `comp/variant/stage` | Glob/substring match on task names; supports `{a,b}` brace expansion. |
| `... X continue` | Trailing `continue` turns the preceding selector `X` into a "from `X` to the end" anchor. Any earlier selectors are selected normally. |

Multiple selectors can be combined; the union of matched tasks runs in task
order. Task names take the form `component/variant/stage` for build tasks
(e.g. `comgr/default/cmake`, `llvm_runtimes_standalone/asan/build`). The variant
segment is dropped (giving `component/stage`) for:

- the config-less init/fini tasks (`precheck`, `patch`, `unpatch` — e.g.
  `comgr/patch`), and
- components whose only advertised config is `default` (e.g. `prereq/build`,
  `rocminfo/cmake`).

Putting the variant in the middle makes it easy to glob a whole variant across
components, e.g. `*/asan/*`.

`continue` is a **trailing** keyword: it must be the last argument, and it
applies to the selector immediately before it. `X` may be a task number or a
task name; if it matches several tasks (e.g. a component name), continuation
starts from its first task.

Examples:

```bash
./aomp_build.py 5                      # just task 5
./aomp_build.py 5--12                  # tasks 5 through 12
./aomp_build.py 30 continue            # from task 30 to the end
./aomp_build.py rocr/default/cmake continue   # from rocr's cmake task onward
./aomp_build.py comgr continue         # from comgr's first task to the end
./aomp_build.py project/default/build comgr continue  # project's build, then comgr onward
./aomp_build.py 'comgr/*'              # every comgr task
./aomp_build.py '*/install'            # every install task (any component/variant)
./aomp_build.py '*/asan/*'             # every asan-variant task
./aomp_build.py '{comgr,rocr}/*/build' # comgr build + rocr build
```

> Tip: quote selectors containing `*`, `?`, or `{}` so your shell doesn't try
> to expand them first.

---

## Build variants

Each component advertises one or more *configs* (variants) through its
`list_configs` command. There are two common styles:

- **Style A** (most components): always offer a plain `default` plus opt-in
  variants such as `asan` and `debug`.
- **Style B** (e.g. `llvm_runtimes_standalone`): derive their config set from
  the environment (`AOMP_BUILD_SANITIZER`, `AOMP_BUILD_PERF`,
  `AOMP_BUILD_DEBUG`) and have **no** plain `default` — their default runtime
  libraries are produced by the `project`/LLVM build itself. The advertised set
  is the extra instrumented variants (e.g. `asan`, `perf`, `perf+asan`,
  `debug`, plus `*-devicertl` device-runtime passes).

The `default` config is the baseline an installable build needs, so it is
**always built when a component offers it**.

Selection policy:

- **No `--variant`:** build **every** advertised config for every component
  (the full build — `default` plus all variants such as `asan`/`debug`/`perf`,
  including the runtimes matrix).
- **`--variant` (any explicit value):** build `default` (where offered) **plus**
  the requested variants each component advertises. So `--variant debug` means
  `default,debug` everywhere: default-only components (e.g. `comgr`) still build
  their `default`, components offering `debug` add it, and components offering
  no `default` (e.g. the runtimes) build just their `debug`. A component that
  offers **neither** `default` **nor** any requested variant is **skipped
  entirely** (no tasks, not even `precheck`/`patch`). In particular,
  `--variant default` builds only the `default` config of each component and
  therefore **skips `llvm_runtimes_standalone`** (its default runtime libraries
  are produced by the `project`/LLVM build itself).

Forms (all combinable, comma-separated and/or repeated):

| Form | Meaning |
|------|---------|
| `--variant cfg` | Apply `cfg` globally (every component that advertises it). |
| `--variant cfg1,cfg2` | Multiple global variants (e.g. `debug,asan`). |
| `--variant comp=cfg` | Override one component only. |
| `--variant comp1=cfg1,comp2=cfg2` | Per-component overrides in one value. |

When only per-component overrides are given (no global value), components not
named build their normal default.

Examples:

```bash
./aomp_build.py --variant default          # default everywhere; skips the runtimes
./aomp_build.py --variant debug,asan        # default + debug + asan (where offered)
./aomp_build.py --variant asan list         # default + asan
./aomp_build.py --variant llvm_runtimes_standalone=perf list
./aomp_build.py --variant rocr=debug,llvm_runtimes_standalone=asan
```

---

## Components and features

The default component set is the `request:` stanza of the config — the
standalone x86_64 AOMP build from `build_aomp.sh`, **omitting the deprecated
classic Flang stack** (`llvm-classic`, `flang-classic`, `pgmath`, `flang`,
`flang_runtime`) as well as the optional debug/hipfort components and the ROCm
math libraries. Re-enable classic Flang with `--add flang`.

Adjust it with `--add` / `--remove`, which accept **component** names or
**feature** names. Each option is repeatable and also accepts a comma-separated
list, so `--add rocmlibs,debug` and `--add rocmlibs --add debug` are
equivalent. Features defined in the default config:

| Feature | Expands to |
|---------|-----------|
| `flang` | `llvm-classic`, `flang-classic`, `pgmath`, `flang`, `flang_runtime` (deprecated classic Flang; not in the default set) |
| `hip` | `hipcc`, `hipamd`, `hipify` |
| `debug` | `rocdbgapi`, `rocgdb` |
| `profiler` | `rocprofiler-register`, `rocprofiler-sdk` |
| `rocmlibs` | `rocm-cmake`, `rocBLAS`, `rocPRIM`, `rocSPARSE`, `rocSOLVER`, `hipBLAS-common`, `hipBLAS`, `rocRAND`, `hipRAND`, `rccl`, `half`, `hipSOLVER` |

Resolution rules:

- `--add` pulls in the named component(s)/feature(s) **and their transitive
  dependencies**.
- `--remove` drops the named component(s)/feature(s) **and anything that
  depends on them** (the removal cascades).

```bash
./aomp_build.py --add rocmlibs --components       # core AOMP + ROCm libraries
./aomp_build.py --remove flang --components       # drop the whole Flang group
./aomp_build.py --add rocgdb --components          # adds rocdbgapi too (dependency)
./aomp_build.py --add rocmlibs,debug --components  # comma-separated list
```

---

## The version manifest

The manifest captures the exact git state of every component's source repo, so
a build can be reproduced later.

Export:

```bash
./aomp_build.py --export-manifest                       # default path
./aomp_build.py --add rocmlibs --export-manifest m.json # explicit path + set
```

The JSON records, per component that has a git source:

```json
{
  "generated": "2026-06-17T10:00:00",
  "config": "aomp",
  "order": ["prereq", "project", "..."],
  "components": {
    "project": {
      "sha": "…",
      "repo": "https://github.com/…",
      "branch": "amd-staging",
      "dirty": false
    }
  }
}
```

Import (a pre-step before building):

```bash
./aomp_build.py --import-manifest m.json [selectors...]
```

- For each component in the manifest that is also in the resolved set, the
  recorded SHA is checked out (`git checkout <sha>`).
- **Safety:** if any target repo has local modifications, the import is
  **refused** entirely (nothing is checked out) and the dirty repos are listed.
- Components without a git source (e.g. `prereq`) are skipped.

Manifests live under `<BUILD_DIR>/manifests/` by default
(`BUILD_DIR` = `$BUILD_AOMP`, normally `$AOMP_REPOS`).

---

## Logs

Each executed task writes a numbered log:

```
<log-dir>/NNN-component-variant-stage.log
```

(config-less init/fini tasks and default-only components are
`NNN-component-stage.log`.)

where `NNN` is the global task number (matching `list`) and `<log-dir>`
defaults to `<BUILD_DIR>/aomp_build_logs`. Each log begins with the task
number, the exact command, and a start timestamp, and ends with an end
timestamp and the return code.

On failure, the orchestrator prints the failing task and tails the log to
stderr, then stops (non-zero exit). Re-run with `<failed-task> continue` after
fixing the problem.

---

## User workflows

### Full build from scratch

```bash
cd $AOMP_REPOS/aomp/bin
./aomp_build.py            # builds the default component set in order
```

### Inspect before building

```bash
./aomp_build.py --components   # which components, in what order
./aomp_build.py list           # every task, numbered
./aomp_build.py -n             # full dry-run (commands + log paths)
```

### Rebuild a single component

```bash
./aomp_build.py 'comgr/*'              # all comgr tasks (incremental: no clean)
./aomp_build.py -C 'comgr/*'           # force a clean rebuild of comgr
./aomp_build.py comgr/default/build    # just re-run comgr's build step
```

### Resume after a failure

A task fails; its log is tailed to your terminal. Fix the issue, then resume
with a trailing `continue`:

```bash
./aomp_build.py comgr/default/cmake continue   # re-run from comgr's cmake onward
# or by number, using the value shown in `list`:
./aomp_build.py 30 continue
```

### Build extra component sets

```bash
./aomp_build.py --add rocmlibs                 # core AOMP + ROCm math libs
./aomp_build.py --add rocmlibs 'rocBLAS/*'     # only rocBLAS tasks (deps still resolved)
./aomp_build.py --add debug                    # add rocdbgapi + rocgdb
```

### Build a sanitizer / perf / debug variant

```bash
./aomp_build.py --variant asan                 # asan everywhere it's offered
./aomp_build.py --variant llvm_runtimes_standalone=debug
```

### Tune the build environment

```bash
./aomp_build.py -j 32 --ninja --ccache         # threads + ninja + ccache
./aomp_build.py --gfx "gfx90a;gfx942"          # specific GPU targets
```

### Reproduce an earlier build

```bash
# On the reference machine:
./aomp_build.py --export-manifest good.json
# Later / elsewhere (repos must be clean):
./aomp_build.py --import-manifest good.json
```

---

## Internals

### The taskified component interface

Every component script (`build_<name>.sh`) speaks a common interface provided
by `command_dispatcher` in [`bin/aomp_utils`](aomp_utils). The orchestrator
relies on exactly these commands:

| Command | Purpose |
|---------|---------|
| `list_configs` | Print the configs/variants this component offers, one per line. |
| `list` | Print every task as `task_<action> [cfg]`, one per line (init tasks, then per-config tasks, then fini tasks). |
| `task_<action> [cfg]` | Run a single task for a given config. |
| `show_src_dir` | Print the component's git source directory (for the manifest). |
| `show_build_dir <cfg>` / `show_install_dir <cfg>` | Print build/install directories (informational). |

Because `list` emits each line already in `task_<action> [cfg]` form, the
orchestrator can pass a line straight back to the script to execute it.

`build_supp.sh` (and its `build_prereq.sh` symlink) is modeled as a single
coarse component: one `default` config, a `task_build` that runs the existing
"build all prerequisite/supplemental components" logic, and a no-op
`task_install`. It still supports its legacy direct invocation
(`build_supp.sh openmpi`, no-arg, `install`, `-h`).

### The CUDF config format

The config (default [`bin/configs/aomp.cudf`](configs/aomp.cudf)) is a
CUDF-style (Common Upgradeability Description Format) text file. Stanzas are
separated by blank lines; `#` lines are comments. Within a stanza each line is
`key: value`; a line whose first token has no `:` is treated as a continuation
of the previous value (used to wrap long comma-separated lists).

Three stanza kinds:

```
package: comgr            # a buildable component
version: 1                # stanza schema version (always 1)
depends: project, rocr    # components that must build first (a DAG)
x-dir: .                  # bin subdir with build_<name>.sh: "." or "rocmlibs"

feature: flang            # a named alias for a set of components
expands: llvm-classic, flang-classic, pgmath, flang, flang_runtime

request:                  # the default requested build set
install: prereq, project, ...
```

The package list, ordering, and grouping are derived from
[`bin/build_aomp.sh`](build_aomp.sh) (standalone x86_64 build) and
[`bin/rocmlibs/build_rocmlibs.sh`](rocmlibs/build_rocmlibs.sh). Dependencies are
expressed as a DAG; packages are declared in canonical build order so the
topological sort reproduces that order (see below).

The parser is intentionally simple and lives entirely in `parse_cudf()`. It
leaves room for a future external version *solver* (the "CUDF" framing): today
the `request:` set plus `--add`/`--remove` is resolved directly, but the same
config could later feed a real dependency/version solver.

### Resolution pipeline

`resolve_components()` turns the config + `--add`/`--remove` into an ordered
component list:

1. Start from `request.install`.
2. `requested |= expand(--add)` and `removed = expand(--remove)`; features are
   expanded to their component sets via `expand_names()`.
3. `requested -= removed`; unknown names are rejected.
4. **Transitive closure with removal cascade** (fixed-point loop): for each
   component, pull in its `depends`; if a dependency is in the removed set,
   drop the dependent (and add it to the removed set so its own dependents
   cascade too). This is why `--remove comgr` also drops `hipamd`,
   `rocdbgapi`, etc.
5. **Topological sort** (`topo_sort()`): a deterministic Kahn's algorithm where
   ready nodes are always taken in *declaration order*. Because packages are
   declared in canonical `build_aomp.sh` order, the result reproduces that
   canonical order while still honoring real dependency edges. A cycle is a
   hard error.

### Task elaboration

`elaborate_tasks()` expands the ordered component list into a flat task list:

1. For each component, locate its script via `script_path()` (honoring
   `x-dir`).
2. Query `list_configs`; pick the configs to build via `select_variants()`
   (see [Build variants](#build-variants)). If an explicit `--variant` filter
   matches none of the component's configs, the component is skipped entirely.
3. Query `list`; parse each `task_<action> [cfg]` line.
4. Keep a task if either it has no config (init/fini tasks such as `precheck`,
   `patch`, `unpatch` always run) or its config is in the selected set.
5. Drop `clean` tasks unless `-C/--clean` was given (incremental by default).

Each surviving task is recorded as a `Task` (component, action, config, script
path, and the exact `script_args` to pass back). Its display name is
`component/variant/stage`, shortened to `component/stage` when the task has no
config or when the component advertises only the `default` config.

### Execution

`select_tasks()` applies the selector grammar to produce a list of task
indices; `run_tasks()` runs them:

- Creates the log directory.
- For each task, opens `NNN-component-variant-stage.log`, writes a header
  (number, command, start time), runs `bash build_<name>.sh task_<action>
  [cfg]` with stdout+stderr redirected to the log, and writes a footer (end
  time, return code).
- On a non-zero return code, prints the failure and tails the log to stderr,
  then returns that code (stops the run).
- `--dry-run` prints the planned command and log path instead of executing.

### Environment discovery and child environment

- `discover_env()` sources `aomp_utils` + `aomp_common_vars` in a subshell and
  reads back `BUILD_DIR`, `AOMP_REPOS`, and `AOMP_REPO_NAME` — used only to
  locate the default log and manifest directories.
- `build_child_env()` starts from the current environment and overlays the
  values implied by the build-knob flags (`-j`, `--ninja`, `--ccache`,
  `--gfx`, `--build-type`, `--sudo`). This same environment is used for every
  child script invocation (introspection, execution, and git/manifest probes),
  so introspection sees the same configs that will actually be built.

---

## Troubleshooting

**A component shows no tasks in `list`.**
Its `list` produced no `task_*` lines — usually the component self-disables in
the current environment (e.g. `llvm-classic`/`flang-classic` when the classic
LLVM source is absent). Such components emit a warning on **stderr** and exit
cleanly, so they simply contribute nothing. Confirm by running the script
directly: `bash build_<name>.sh list`.

**`unknown component(s): …`.**
A name passed to `--add`/`--remove` (or in the config's `request:`) is neither
a package nor a feature. Check spelling against `--components` and the config.

**`dependency cycle among: …`.**
The `depends:` edges in the config form a cycle. Fix the config.

**Import refuses with "local modifications".**
A target repo is dirty. Commit/stash/clean it, or drop it from the resolved set
(`--remove`), then retry the import.

**A task fails.**
Read the tailed log (path printed on failure, under the log dir). After fixing,
resume with `<task> continue`.

---

## Extending

- **Add a component:** create `build_<name>.sh` implementing the
  [task interface](#the-taskified-component-interface), then add a `package:`
  stanza (with `depends:` and `x-dir:`) to the config. Declare it in canonical
  build position so the topological sort places it correctly.
- **Add a feature:** add a `feature:` stanza with an `expands:` list.
- **Change the default set:** edit the `request:` stanza, or keep the config
  fixed and use `--add`/`--remove` at the command line.
- **Use a different config:** `-c /path/to/other.cudf`. The manifest's default
  filename follows the config's basename.
