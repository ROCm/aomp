#!/usr/bin/env python3
"""aomp_build.py - unified, introspectable AOMP build orchestrator.

This driver pulls the taskified per-component build scripts (build_<name>.sh,
each speaking the command_dispatcher interface from aomp_utils) into a single
workflow:

  * resolve a component set from a CUDF-style config (configs/aomp.cudf),
    honoring --add/--remove of components and feature aliases;
  * topologically order the components (deterministic, declaration-order
    tie-break, which reproduces the canonical build_aomp.sh ordering);
  * elaborate fine-grained tasks by querying each component's `list`,
    filtered by the selected build variant(s);
  * list those tasks, or run them individually, by number/range, from a
    point onward (`continue`), by glob, or all at once - mirroring the
    `amd-build` selector grammar;
  * write per-task numbered logs and tail the log on failure;
  * export/import a JSON version manifest (per-component git fingerprint) for
    reproducible builds.

Stdlib only; intended to run under the same environment as the build scripts.
"""

from __future__ import annotations

import argparse
import datetime
import fnmatch
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field

BIN_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.join(BIN_DIR, "configs", "aomp.cudf")

# Child build scripts run in an isolated environment built from scratch by this
# orchestrator, so that what gets built is under the orchestrator's control and
# not at the mercy of whatever the caller happened to have exported. Only the
# variables below are passed through unchanged: these are identity / locale /
# terminal settings that affect *how* things look or *who* git acts as, not
# *what* gets compiled. Everything else (CC, CXX, LD_LIBRARY_PATH, PKG_CONFIG_*,
# AOMP_*, ROCM_*, ...) is dropped unless the orchestrator sets it explicitly or
# the user opts in with --pass-env. Any variable named LC_* is also passed
# through (locale categories).
ENV_PASSTHROUGH = (
    "HOME", "USER", "LOGNAME", "SHELL", "TERM",
    "LANG", "LANGUAGE", "TZ", "TMPDIR",
    "DISPLAY", "XAUTHORITY",
    "SSH_AUTH_SOCK", "SSH_CONNECTION", "SSH_CLIENT", "SSH_TTY",
)

# Deterministic PATH handed to child build scripts. Standard system locations
# only, so tool resolution is predictable and not shadowed by whatever the
# caller put earlier on their PATH (e.g. a Homebrew pkg-config that cannot see
# the system .pc files). Override with --inherit-path to use the caller's PATH.
DEFAULT_CHILD_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"


# --------------------------------------------------------------------------- #
# Config model + CUDF parser
# --------------------------------------------------------------------------- #
@dataclass
class Package:
    name: str
    depends: list[str] = field(default_factory=list)
    xdir: str = "."
    order: int = 0  # declaration order, used as topo-sort tie-break


@dataclass
class Config:
    packages: dict[str, Package] = field(default_factory=dict)
    features: dict[str, list[str]] = field(default_factory=dict)
    request: list[str] = field(default_factory=list)


def _split_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_cudf(path: str) -> Config:
    """Parse a CUDF-style config file into a Config.

    Stanzas are separated by blank lines; '#' lines are comments. A line whose
    first token has no trailing ':' is treated as a continuation of the
    previous key's value (used to wrap long comma-separated lists).
    """
    cfg = Config()
    order = 0

    try:
        with open(path, encoding="utf-8") as handle:
            raw_lines = handle.readlines()
    except OSError as exc:
        sys.exit(f"aomp_build: cannot read config '{path}': {exc}")

    stanza: dict[str, str] = {}

    def flush(stanza: dict[str, str]) -> None:
        nonlocal order
        if not stanza:
            return
        if "package" in stanza:
            name = stanza["package"]
            pkg = Package(
                name=name,
                depends=_split_list(stanza.get("depends", "")),
                xdir=stanza.get("x-dir", ".") or ".",
                order=order,
            )
            cfg.packages[name] = pkg
            order += 1
        elif "feature" in stanza:
            cfg.features[stanza["feature"]] = _split_list(stanza.get("expands", ""))
        elif "request" in stanza or "install" in stanza:
            cfg.request = _split_list(stanza.get("install", ""))

    last_key: str | None = None
    for line in raw_lines:
        stripped = line.strip()
        if not stripped:
            flush(stanza)
            stanza = {}
            last_key = None
            continue
        if stripped.startswith("#"):
            continue
        # Continuation line: no "key:" at the start.
        if ":" not in stripped.split(" ", 1)[0] and last_key is not None:
            stanza[last_key] = (stanza.get(last_key, "") + " " + stripped).strip()
            continue
        key, _, value = stripped.partition(":")
        key = key.strip()
        stanza[key] = value.strip()
        last_key = key
    flush(stanza)

    if not cfg.packages:
        sys.exit(f"aomp_build: no packages found in config '{path}'")
    return cfg


# --------------------------------------------------------------------------- #
# Feature expansion + dependency resolution
# --------------------------------------------------------------------------- #
def expand_names(names: list[str], cfg: Config) -> set[str]:
    """Expand component/feature names into a set of components.

    Each entry may itself be a comma-separated list (so `--add a,b` and
    `--add a --add b` are equivalent), and any entry naming a feature is
    expanded to that feature's components.
    """
    result: set[str] = set()
    for entry in names:
        for name in (n.strip() for n in entry.split(",")):
            if not name:
                continue
            if name in cfg.features:
                result.update(cfg.features[name])
            else:
                result.add(name)
    return result


def resolve_components(
    cfg: Config, adds: list[str], removes: list[str]
) -> list[str]:
    """Resolve the final, topologically ordered component list.

    request +/- (features|components), then transitive dependency closure with
    removal cascade: removing a component also drops anything that depends on
    it (directly or transitively).
    """
    requested = set(cfg.request)
    requested |= expand_names(adds, cfg)
    removed = expand_names(removes, cfg)
    requested -= removed

    unknown = {n for n in requested | removed if n not in cfg.packages}
    if unknown:
        sys.exit(
            "aomp_build: unknown component(s): " + ", ".join(sorted(unknown))
        )

    # Transitive dependency closure with removal cascade.
    final = set(requested)
    changed = True
    while changed:
        changed = False
        for comp in list(final):
            for dep in cfg.packages[comp].depends:
                if dep in removed:
                    final.discard(comp)
                    removed.add(comp)
                    changed = True
                    break
                if dep not in final:
                    final.add(dep)
                    changed = True

    return topo_sort(cfg, final)


def topo_sort(cfg: Config, comps: set[str]) -> list[str]:
    """Deterministic Kahn topological sort with declaration-order tie-break."""
    indeg = {c: 0 for c in comps}
    succ: dict[str, list[str]] = {c: [] for c in comps}
    for comp in comps:
        for dep in cfg.packages[comp].depends:
            if dep in comps:
                indeg[comp] += 1
                succ[dep].append(comp)

    ready = sorted((c for c in comps if indeg[c] == 0),
                   key=lambda c: cfg.packages[c].order)
    ordered: list[str] = []
    while ready:
        comp = ready.pop(0)
        ordered.append(comp)
        for nxt in succ[comp]:
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                ready.append(nxt)
        ready.sort(key=lambda c: cfg.packages[c].order)

    if len(ordered) != len(comps):
        cycle = comps - set(ordered)
        sys.exit("aomp_build: dependency cycle among: " + ", ".join(sorted(cycle)))
    return ordered


# --------------------------------------------------------------------------- #
# Build-script invocation helpers
# --------------------------------------------------------------------------- #
def script_path(cfg: Config, comp: str) -> str:
    xdir = cfg.packages[comp].xdir
    if xdir in (".", "", None):
        return os.path.join(BIN_DIR, f"build_{comp}.sh")
    return os.path.join(BIN_DIR, xdir, f"build_{comp}.sh")


def run_script_capture(path: str, args: list[str], env: dict[str, str]) -> str:
    """Run a build script and return stdout (used for introspection)."""
    proc = subprocess.run(
        ["bash", path, *args],
        capture_output=True,
        text=True,
        env=env,
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        sys.exit(
            f"aomp_build: '{os.path.basename(path)} {' '.join(args)}' "
            f"failed (rc={proc.returncode})"
        )
    return proc.stdout


def discover_env(env: dict[str, str] | None = None) -> dict[str, str]:
    """Source aomp_utils + aomp_common_vars to learn BUILD_DIR/AOMP_REPOS.

    The same isolated child environment used for the build is passed in, so the
    discovered values reflect any -i/-b/-p directory overrides (e.g. BUILD_DIR
    follows --build, used for the default log/manifest locations).
    """
    snippet = (
        f'. "{BIN_DIR}/aomp_utils" >/dev/null 2>&1; '
        f'. "{BIN_DIR}/aomp_common_vars" >/dev/null 2>&1; '
        'printf "%s\\n" "$BUILD_DIR" "$AOMP_REPOS" "$AOMP_REPO_NAME" '
        '"$AOMP_INSTALL_DIR" "$AOMP"'
    )
    proc = subprocess.run(
        ["bash", "-c", snippet], capture_output=True, text=True, env=env
    )
    out = proc.stdout.splitlines()
    out += [""] * (5 - len(out))
    return {
        "BUILD_DIR": out[0] or os.path.join(os.path.expanduser("~"), "git", "aomp"),
        "AOMP_REPOS": out[1] or os.path.join(os.path.expanduser("~"), "git", "aomp"),
        "AOMP_REPO_NAME": out[2] or "aomp",
        # The versioned install dir (the real target the scripts symlink to) and
        # the symlink itself; used by -C/--clean to wipe a stale installation.
        "AOMP_INSTALL_DIR": out[3],
        "AOMP": out[4],
    }


# --------------------------------------------------------------------------- #
# Task elaboration
# --------------------------------------------------------------------------- #
@dataclass
class Task:
    comp: str
    action: str          # e.g. "cmake" (without the "task_" prefix)
    cfgname: str | None  # build config/variant, e.g. "default" / "asan"
    script: str          # path to build_<comp>.sh
    script_args: list[str]  # args to pass back to the script
    single_config: bool = False  # component advertises only "default"
    # Builtin (orchestrator-run) tasks have no backing script. Currently only
    # "install_clean": wipe the install dir. `targets` then holds the paths it
    # operates on ([install_dir, symlink]).
    builtin: str | None = None
    targets: list[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        # "component/variant/stage" for config-bearing tasks. The variant
        # segment is dropped (-> "component/stage") for the config-less
        # init/fini tasks (precheck/patch/unpatch) and for components whose only
        # advertised config is "default".
        if self.cfgname and not self.single_config:
            return f"{self.comp}/{self.cfgname}/{self.action}"
        return f"{self.comp}/{self.action}"


def component_configs(script: str, env: dict[str, str]) -> list[str]:
    out = run_script_capture(script, ["list_configs"], env)
    return [line.strip() for line in out.splitlines() if line.strip()]


def select_variants(
    comp: str, available: list[str], global_variants: list[str],
    per_comp: dict[str, list[str]],
) -> list[str]:
    """Decide which build configs to elaborate for a component.

    Components advertise their configs via `list_configs`. Two styles exist:
    style-A always offers a plain "default" plus opt-in variants (asan/debug);
    style-B (e.g. the runtimes) derives its config set from the environment and
    has no "default" because its default build is produced by the compiler
    (project) build itself.

    The "default" config is the baseline an installable build needs, so it is
    always built when the component offers it.

    Selection:
      * No --variant anywhere: build *every* advertised config (the full build:
        default plus all variants the component offers).
      * --variant given: build "default" (when offered) plus the requested
        variants the component advertises. Requested variants apply globally,
        or to a single component via "comp=cfg" (which overrides the global
        list for that component). A component that offers neither "default" nor
        any requested variant yields an empty list and is skipped entirely - so
        `--variant default` builds just the default config and skips components
        that have none (e.g. the runtimes, built by the project build).
    """
    if not global_variants and not per_comp:
        # No filter at all: build everything the component advertises.
        return list(available)

    requested = per_comp[comp] if comp in per_comp else global_variants

    wanted: list[str] = []
    if "default" in available:
        wanted.append("default")
    for variant in requested:
        if variant in available and variant not in wanted:
            wanted.append(variant)
    return wanted


def elaborate_tasks(
    cfg: Config, components: list[str], env: dict[str, str],
    global_variants: list[str], per_comp_variants: dict[str, list[str]],
) -> list[Task]:
    tasks: list[Task] = []
    for comp in components:
        script = script_path(cfg, comp)
        if not os.path.exists(script):
            sys.exit(f"aomp_build: missing build script for '{comp}': {script}")
        available = component_configs(script, env)
        wanted = select_variants(
            comp, available, global_variants, per_comp_variants
        )
        # An explicit --variant filter that matches none of this component's
        # configs skips the component outright (no init/fini tasks either).
        if available and not wanted:
            continue
        # Components whose only advertised config is "default" use the short
        # two-element task name (comp/stage) instead of comp/default/stage.
        single_config = available == ["default"]
        listing = run_script_capture(script, ["list"], env).splitlines()
        for line in listing:
            toks = line.split()
            if not toks:
                continue
            action_tok = toks[0]
            if not action_tok.startswith("task_"):
                continue
            action = action_tok[len("task_"):]
            taskcfg = toks[1] if len(toks) > 1 else None
            # Variant filter: config-bearing tasks must match a wanted config;
            # config-less tasks (init/fini such as patch/unpatch) always run.
            if taskcfg is not None and taskcfg not in wanted:
                continue
            tasks.append(
                Task(
                    comp=comp,
                    action=action,
                    cfgname=taskcfg,
                    script=script,
                    script_args=toks,
                    single_config=single_config,
                )
            )
    return tasks


# --------------------------------------------------------------------------- #
# Selector grammar (amd-build style)
# --------------------------------------------------------------------------- #
def brace_expand(token: str) -> list[str]:
    """Minimal brace expansion: a{b,c}d -> [abd, acd]. Single level."""
    match = re.search(r"\{([^{}]*)\}", token)
    if not match:
        return [token]
    pre, post = token[: match.start()], token[match.end():]
    out: list[str] = []
    for part in match.group(1).split(","):
        out.extend(brace_expand(pre + part + post))
    return out


def select_tasks(tasks: list[Task], selectors: list[str]) -> list[int]:
    """Return the indices (into tasks) selected by the positional selectors.

    Grammar (mirrors amd-build, with a trailing `continue`):
      * no selectors            -> all tasks
      * N                       -> task number N (1-based)
      * N--M                    -> inclusive range
      * glob (with {a,b} braces)-> substring/glob match on
                                   "comp/variant/stage" (or "comp/stage")
      * ... X continue          -> trailing `continue` turns the preceding
                                   selector X into a "from X to the end" anchor;
                                   any earlier selectors are selected normally.
                                   e.g. `comp1 comp2 continue` runs comp1's
                                   tasks then everything from comp2 onward.
    """
    if not selectors:
        return list(range(len(tasks)))

    names = [t.name for t in tasks]
    selectors = list(selectors)

    # `continue` is a trailing modifier: it must be the final token and turns
    # the selector immediately before it into a continue-to-the-end anchor.
    continue_from_last = False
    if selectors and selectors[-1] == "continue":
        selectors.pop()
        if not selectors:
            sys.exit(
                "aomp_build: trailing 'continue' requires a preceding task "
                "number or name (e.g. 'comp2 continue')"
            )
        continue_from_last = True
    if "continue" in selectors:
        sys.exit(
            "aomp_build: 'continue' must be the last selector "
            "(e.g. 'comp1 comp2 continue' builds comp1 then continues from comp2)"
        )

    selected: list[int] = []

    def add(idx: int) -> None:
        if 0 <= idx < len(tasks) and idx not in selected:
            selected.append(idx)

    def match_indices(sel: str) -> list[int]:
        """Indices matched by a single (non-continue) selector."""
        if sel.isdigit():
            return [int(sel) - 1]
        if "--" in sel:
            lo_s, hi_s = sel.split("--", 1)
            lo, hi = match_point(lo_s), match_point(hi_s)
            if lo > hi:
                lo, hi = hi, lo
            return list(range(lo, hi + 1))
        out: list[int] = []
        for pat in brace_expand(sel):
            for idx, nm in enumerate(names):
                if fnmatch.fnmatch(nm, pat) or pat == nm or pat in nm:
                    out.append(idx)
        if not out:
            sys.exit(f"aomp_build: selector '{sel}' matched no task")
        return out

    def match_point(token: str) -> int:
        if token.isdigit():
            return int(token) - 1
        for i, nm in enumerate(names):
            if token == nm or fnmatch.fnmatch(nm, token) or token in nm:
                return i
        sys.exit(f"aomp_build: selector '{token}' matched no task")

    last = len(selectors) - 1
    for i, sel in enumerate(selectors):
        if continue_from_last and i == last:
            # Anchor: from this selector's first matched task to the end.
            start = min(match_indices(sel))
            for idx in range(start, len(tasks)):
                add(idx)
        else:
            for idx in match_indices(sel):
                add(idx)

    selected.sort()
    return selected


# --------------------------------------------------------------------------- #
# Execution
# --------------------------------------------------------------------------- #
def tail_file(path: str, n: int = 40) -> str:
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            lines = handle.readlines()
    except OSError:
        return ""
    return "".join(lines[-n:])


def stamp_path(stamp_dir: str, task: Task, kind: str) -> str:
    """Path of a task's stamp file (keyed by its stable name). kind is one of
    'start' (written when the task begins) or 'done' (written on success)."""
    return os.path.join(stamp_dir, task.name.replace("/", "-") + "." + kind)


def task_state(stamp_dir: str | None, task: Task) -> str:
    """Completion state inferred from the stamps:

      'done'       -> the task finished (a 'done' stamp is present)
      'incomplete' -> it started but did not finish ('start' only)
      'none'       -> no stamp at all
    """
    if not stamp_dir:
        return "none"
    if os.path.exists(stamp_path(stamp_dir, task, "done")):
        return "done"
    if os.path.exists(stamp_path(stamp_dir, task, "start")):
        return "incomplete"
    return "none"


def render_mark(state: str) -> str:
    """A mark for a task state: green check (done), red cross (incomplete),
    or a blank of the same width (none)."""
    tty = sys.stdout.isatty()
    if state == "done":
        return "\033[32m\u2713\033[0m" if tty else "\u2713"
    if state == "incomplete":
        return "\033[31m\u2717\033[0m" if tty else "\u2717"
    return " "


def clear_stamps_from(stamp_dir: str, tasks: list[Task], start_index: int) -> None:
    """Remove both stamps for every task at index >= start_index."""
    for task in tasks[start_index:]:
        for kind in ("start", "done"):
            try:
                os.remove(stamp_path(stamp_dir, task, kind))
            except FileNotFoundError:
                pass


def make_install_clean_task(env_info: dict[str, str]) -> Task:
    """The -C/--clean pseudo-task: wipe the install directory (the versioned
    symlink *target*, not just the symlink). Inserted at the front of the task
    list so a stale/partially-installed tree is removed before anything builds.
    """
    install_dir = env_info.get("AOMP_INSTALL_DIR", "")
    symlink = env_info.get("AOMP", "")
    return Task(
        comp="install", action="clean", cfgname=None,
        script="", script_args=[], single_config=True,
        builtin="install_clean", targets=[install_dir, symlink],
    )


def run_install_clean(targets: list[str], env: dict[str, str], log) -> int:
    """Wipe the install dir (targets[0]) and drop the symlink (targets[1]) if it
    is a distinct symlink. Honors SUDO (the install may be root-owned)."""
    install_dir = targets[0] if targets else ""
    symlink = targets[1] if len(targets) > 1 else ""

    abs_install = os.path.abspath(install_dir) if install_dir else ""
    if not abs_install or abs_install in ("/", os.path.abspath(os.path.expanduser("~"))):
        log.write(f"ERROR: refusing to wipe unsafe install dir '{install_dir}'\n")
        log.flush()
        return 1

    sudo = env.get("SUDO", "")
    prefix = ["sudo"] if sudo in ("set", "yes", "YES") else []

    def run(cmd: list[str]) -> int:
        log.write(" ".join(cmd) + "\n")
        log.flush()
        return subprocess.run(
            cmd, stdout=log, stderr=subprocess.STDOUT, env=env
        ).returncode

    rc = run(prefix + ["rm", "-rf", "--", install_dir])
    # If the symlink is distinct from the target, remove the dangling link too.
    if rc == 0 and symlink and symlink != install_dir and os.path.islink(symlink):
        rc = run(prefix + ["rm", "-f", "--", symlink])
    return rc


def run_tasks(
    tasks: list[Task], indices: list[int], env: dict[str, str], log_dir: str,
    dry_run: bool, build_type_global: str | None = None,
    build_type_per_comp: dict[str, str] | None = None,
    log_base: str | None = None, stamp_dir: str | None = None,
) -> int:
    build_type_per_comp = build_type_per_comp or {}
    os.makedirs(log_dir, exist_ok=True)
    # Any run clears stamps from the lowest task index onward (a full run thus
    # resets everything), so the stamps reflect only the current build attempt.
    if stamp_dir and not dry_run:
        os.makedirs(stamp_dir, exist_ok=True)
        if indices:
            clear_stamps_from(stamp_dir, tasks, min(indices))
    # The progress counter is relative to the whole build: num is the task's
    # absolute position (1-based) in the full elaborated task list and total is
    # the full count, so a selected subset still reports its real task numbers.
    total = len(tasks)
    width = len(str(total))
    for idx in indices:
        task = tasks[idx]
        num = idx + 1
        safe = task.name.replace("/", "-")
        log_path = os.path.join(log_dir, f"{num:03d}-{safe}.log")
        if task.builtin == "install_clean":
            cmd = ["rm", "-rf", *(t for t in task.targets if t)]
        else:
            cmd = ["bash", task.script, *task.script_args]
        # Per-component BUILD_TYPE override (per-component wins over global).
        build_type = build_type_per_comp.get(task.comp, build_type_global)
        task_env = env
        if build_type:
            task_env = dict(env)
            task_env["BUILD_TYPE"] = build_type
        header = f"[{num:0{width}d}/{total}] {task.name}"
        bt_note = f"  BUILD_TYPE={build_type}" if build_type else ""
        # Show the log path relative to the build root (where logs live) for
        # brevity; fall back to the absolute path if it lies elsewhere.
        rel_log = log_path
        if log_base:
            candidate = os.path.relpath(log_path, log_base)
            if not candidate.startswith(".."):
                rel_log = candidate
        if dry_run:
            print(f"{header}\n    {' '.join(cmd)}{bt_note}  > {rel_log}")
            continue
        print(f"{header}{bt_note} -> {rel_log}", flush=True)
        start = datetime.datetime.now()
        # Mark the task as started (start stamp without a done stamp == an
        # incomplete/failed build until the done stamp is written below).
        if stamp_dir:
            with open(stamp_path(stamp_dir, task, "start"), "w",
                      encoding="utf-8") as st:
                st.write(start.isoformat() + "\n")
        with open(log_path, "w", encoding="utf-8") as log:
            log.write(f"### task {num}: {task.name}\n")
            log.write(f"### command: {' '.join(cmd)}\n")
            if build_type:
                log.write(f"### env: BUILD_TYPE={build_type}\n")
            log.write(f"### start: {start.isoformat()}\n\n")
            log.flush()
            if task.builtin == "install_clean":
                rc = run_install_clean(task.targets, task_env, log)
            else:
                rc = subprocess.run(
                    cmd, stdout=log, stderr=subprocess.STDOUT, env=task_env
                ).returncode
            end = datetime.datetime.now()
            log.write(f"\n### end: {end.isoformat()} (rc={rc})\n")
        if rc != 0:
            print(
                f"\naomp_build: FAILED task {num} ({task.name}), rc={rc}",
                file=sys.stderr,
            )
            print(f"--- tail of {log_path} ---", file=sys.stderr)
            print(tail_file(log_path), file=sys.stderr)
            return rc
        # Mark the task complete.
        if stamp_dir:
            with open(stamp_path(stamp_dir, task, "done"), "w",
                      encoding="utf-8") as st:
                st.write(end.isoformat() + "\n")
    return 0


# --------------------------------------------------------------------------- #
# Version manifest (git fingerprint)
# --------------------------------------------------------------------------- #
# Repositories consumed by component builds (e.g. via LLVM_EXTERNAL_PROJECTS)
# that are not standalone components. Paths are relative to AOMP_REPOS. These
# are recorded under the manifest "externals" section and restored on import
# because they are tightly coupled to the LLVM/comgr toolchain and are a
# frequent source of build breakage, so their exact versions matter.
EXTERNAL_REPOS = {
    "SPIRV-LLVM-Translator": "SPIRV-LLVM-Translator",
}

# Components whose source must never be rolled back on import: their checkout
# is meant to track HEAD. "extras" is the AOMP build-scripts repo (this very
# tree) -- pinning it would change the build logic mid-flight, and the scripts
# are intended to (eventually) build arbitrary AOMP/ROCm versions.
FLOATING_COMPONENTS = {"extras"}


def _git(src: str, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", src, *args], capture_output=True, text=True
    )
    return proc.stdout.strip() if proc.returncode == 0 else ""


def git_toplevel(src: str) -> str:
    """Absolute path of the git repository containing `src` (or "")."""
    if not src or not os.path.isdir(src):
        return ""
    return _git(src, "rev-parse", "--show-toplevel")


def git_facts(src: str) -> dict | None:
    """Git fingerprint of the repository containing `src`.

    `src` may be a subdirectory of its repository: the LLVM project, comgr,
    hipcc and the standalone runtimes all build from different subdirs of the
    single llvm-project checkout. We resolve the enclosing repository root via
    `git rev-parse --show-toplevel` so each such component is recorded, and
    note the build subdir (relative to that root) when it is not the root.
    """
    top = git_toplevel(src)
    if not top:
        return None

    facts = {
        "sha": _git(src, "rev-parse", "HEAD"),
        "repo": _git(src, "config", "--get", "remote.origin.url"),
        "branch": _git(src, "rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(_git(src, "status", "--porcelain")),
    }
    # Build subdir relative to the repo root (handles src being a subdir of a
    # shared repo, e.g. llvm-project/{llvm,amd/comgr,amd/hipcc,runtimes}).
    # `--show-prefix` is symlink-safe, unlike relpath against --show-toplevel.
    subdir = _git(src, "rev-parse", "--show-prefix").rstrip("/")
    if subdir:
        facts["subdir"] = subdir
    return facts


def component_src_dir(cfg: Config, comp: str, env: dict[str, str]) -> str:
    script = script_path(cfg, comp)
    if not os.path.exists(script):
        return ""
    return run_script_capture(script, ["show_src_dir"], env).strip()


def export_manifest(
    cfg: Config, components: list[str], env: dict[str, str], path: str,
    config_name: str,
) -> None:
    manifest = {
        "generated": datetime.datetime.now().isoformat(),
        "config": config_name,
        "order": components,
        "components": {},
        "externals": {},
    }
    for comp in components:
        src = component_src_dir(cfg, comp, env)
        facts = git_facts(src)
        if facts is not None:
            manifest["components"][comp] = facts

    # External (non-component) repos consumed by the LLVM/comgr toolchain.
    repos = discover_env(env)["AOMP_REPOS"]
    for name, rel in EXTERNAL_REPOS.items():
        facts = git_facts(os.path.join(repos, rel))
        if facts is not None:
            manifest["externals"][name] = facts

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"aomp_build: wrote manifest for {len(manifest['components'])} "
          f"component(s) and {len(manifest['externals'])} external repo(s) "
          f"to {path}")


def import_manifest(
    cfg: Config, components: list[str], env: dict[str, str], path: str,
) -> int:
    try:
        with open(path, encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        sys.exit(f"aomp_build: cannot read manifest '{path}': {exc}")

    recorded = manifest.get("components", {})
    recorded_ext = manifest.get("externals", {})

    # First pass: build a checkout plan and refuse if any target repo is dirty.
    # Several components (project, comgr, hipcc, runtimes) share the single
    # llvm-project checkout, so we dedupe by repository root to avoid repeated
    # (and redundant) checkouts/dirty reports.
    plan: list[tuple[str, str, str]] = []  # (label, src, sha)
    dirty: list[str] = []
    seen_tops: set[str] = set()

    def schedule(label: str, src: str, sha: str) -> None:
        facts = git_facts(src)
        if facts is None:
            print(f"aomp_build: skipping '{label}' (no git source at {src})")
            return
        top = git_toplevel(src)
        if top in seen_tops:
            return
        seen_tops.add(top)
        if facts["dirty"]:
            dirty.append(label)
        plan.append((label, src, sha))

    # Components, in build order. Floating components track HEAD and are never
    # rolled back.
    for comp in components:
        if comp not in recorded:
            continue
        if comp in FLOATING_COMPONENTS:
            print(f"aomp_build: keeping '{comp}' at HEAD (floating; not rolled back)")
            continue
        schedule(comp, component_src_dir(cfg, comp, env), recorded[comp]["sha"])

    # External repos (e.g. SPIRV-LLVM-Translator).
    repos = discover_env(env)["AOMP_REPOS"]
    for name, rel in EXTERNAL_REPOS.items():
        if name not in recorded_ext:
            continue
        schedule(name, os.path.join(repos, rel), recorded_ext[name]["sha"])

    if dirty:
        sys.exit(
            "aomp_build: refusing to import manifest; local modifications in: "
            + ", ".join(dirty)
        )

    # Second pass: check out recorded SHAs.
    for label, src, sha in plan:
        if not sha:
            print(f"aomp_build: skipping '{label}' (no recorded sha)")
            continue
        print(f"aomp_build: checking out {label} @ {sha[:12]} in {src}")
        proc = subprocess.run(["git", "-C", src, "checkout", sha])
        if proc.returncode != 0:
            sys.exit(f"aomp_build: git checkout failed for '{label}'")
    return 0


# --------------------------------------------------------------------------- #
# Environment / CLI flag mapping
# --------------------------------------------------------------------------- #
def parse_scoped_specs(specs: list[str]) -> tuple[list[str], dict[str, list[str]]]:
    """Split scoped option values into global values and per-component overrides.

    Used by both --variant and --build-type. Each value may be a comma-separated
    list, and the option is repeatable, so these are equivalent:
        --variant debug,asan
        --variant debug --variant asan
    An entry of the form "comp=value" applies to a single component; bare
    entries apply globally. Both forms can be mixed in one value, e.g.
        --variant comp1=debug,comp2=asan
    Multiple "comp=value" entries for the same component accumulate, in order.
    """
    global_values: list[str] = []
    per_comp: dict[str, list[str]] = {}
    for spec in specs:
        for entry in spec.split(","):
            entry = entry.strip()
            if not entry:
                continue
            if "=" in entry:
                comp, _, val = entry.partition("=")
                per_comp.setdefault(comp.strip(), []).append(val.strip())
            else:
                global_values.append(entry)
    return global_values, per_comp


def parse_build_type_specs(specs: list[str]) -> tuple[str | None, dict[str, str]]:
    """Resolve --build-type into a single global type and per-component types.

    Same grammar as --variant (global value or 'comp=type', comma-separated
    and/or repeatable), but a build type is a single value per scope, so the
    last value wins if several are given for the same scope.
    """
    global_values, per_comp_lists = parse_scoped_specs(specs)
    global_bt = global_values[-1] if global_values else None
    per_comp_bt = {comp: vals[-1] for comp, vals in per_comp_lists.items()}
    return global_bt, per_comp_bt


def build_child_env(args: argparse.Namespace) -> dict[str, str]:
    """Construct the isolated environment for child build scripts.

    Rather than inheriting the caller's environment wholesale, the env is built
    from scratch: a curated pass-through of identity/locale vars (ENV_PASSTHROUGH
    plus LC_*), a controlled PATH, and the build knobs derived from CLI flags.
    --inherit-path leaks the caller's PATH through; --pass-env leaks named vars.
    """
    src = os.environ
    env: dict[str, str] = {}

    for name in ENV_PASSTHROUGH:
        if name in src:
            env[name] = src[name]
    for name, value in src.items():
        if name.startswith("LC_"):
            env[name] = value

    # PATH: controlled by default, optionally inherited from the caller.
    if args.inherit_path:
        env["PATH"] = src.get("PATH", DEFAULT_CHILD_PATH)
    else:
        env["PATH"] = DEFAULT_CHILD_PATH

    # Escape hatch: explicitly leak named variables for machine-specific needs.
    for name in args.pass_env:
        for var in (n.strip() for n in name.split(",")):
            if var and var in src:
                env[var] = src[var]

    def setenv(name: str, value: str) -> None:
        env[name] = value

    def setdir(name: str, value: str | None) -> None:
        # Directory knobs are made absolute (with ~ expansion) so child scripts
        # resolve them identically regardless of their working directory.
        if value is not None:
            env[name] = os.path.abspath(os.path.expanduser(value))

    setdir("AOMP_REPOS", args.source)
    setdir("AOMP", args.install)
    setdir("BUILD_AOMP", args.build)
    setdir("AOMP_SUPP", args.prereq)

    if args.jobs is not None:
        setenv("AOMP_JOB_THREADS", str(args.jobs))
    if args.ninja is not None:
        setenv("AOMP_USE_NINJA", "1" if args.ninja else "0")
    if args.ccache is not None:
        setenv("AOMP_USE_CCACHE", "1" if args.ccache else "0")
    if args.gfx:
        setenv("GFXLIST", args.gfx)
    # BUILD_TYPE is applied per-component at execution time (see run_tasks),
    # so it is intentionally not set in the shared child environment here.
    if args.sudo:
        setenv("SUDO", "yes")
    return env


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aomp_build.py",
        description="Unified AOMP component build orchestrator.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Selectors (positional, amd-build style):\n"
            "  (none)        run all elaborated tasks\n"
            "  list          print the numbered task list and exit\n"
            "  N             run task number N (1-based)\n"
            "  N--M          run the inclusive range of tasks N..M\n"
            "  comp/variant/stage  glob/substring match (supports {a,b} braces);\n"
            "                  config-less init/fini tasks are comp/stage\n"
            "  continue        on its own, resume from the first task not marked\n"
            "                  complete (by its stamp) through to the end\n"
            "  ... X continue  trailing 'continue' makes the preceding selector\n"
            "                  X a 'from X to the end' anchor, e.g.\n"
            "                  'comp1 comp2 continue' builds comp1 then continues\n"
            "                  from comp2 onward\n"
            "\n"
            "Any run clears completion stamps from the lowest selected task\n"
            "onward; 'list' shows a green check (done) or red cross (incomplete).\n"
        ),
    )
    parser.add_argument("selectors", nargs="*", help="task selector(s); see below")
    parser.add_argument("-c", "--config", default=DEFAULT_CONFIG,
                        help=f"CUDF config file (default: {DEFAULT_CONFIG})")
    # Core directory layout (exported to child build scripts).
    parser.add_argument("-s", "--source", default=None, metavar="DIR",
                        help="source/repo root (AOMP_REPOS) holding the cloned "
                             "component repos. The build dir defaults to it "
                             "unless -b is given. Default: $HOME/git/aomp<ver>")
    parser.add_argument("-i", "--install", default=None, metavar="DIR",
                        help="installation directory root (AOMP); the versioned "
                             "install dir AOMP_<version> derives from it. "
                             "Default: $HOME/rocm/aomp")
    parser.add_argument("-b", "--build", default=None, metavar="DIR",
                        help="directory where builds run / object files go "
                             "(BUILD_AOMP). Default: the repo dir (AOMP_REPOS)")
    parser.add_argument("-p", "--prereq", default=None, metavar="DIR",
                        help="prerequisite/supplemental component root (AOMP_SUPP); "
                             "its build/install subdirs and the prereq cmake "
                             "derive from it. Default: $HOME/local")
    parser.add_argument("--add", action="append", default=[], metavar="NAMES",
                        help="add component(s)/feature(s); comma-separated and/or "
                             "repeatable")
    parser.add_argument("--remove", action="append", default=[], metavar="NAMES",
                        help="remove component(s)/feature(s); comma-separated "
                             "and/or repeatable")
    parser.add_argument("--variant", action="append", default=[], metavar="SPEC",
                        help="variant filter: 'cfg' (global) or 'comp=cfg' "
                             "(per-component). Comma-separated and/or repeatable "
                             "(e.g. 'debug,asan'). 'default' is always built when "
                             "offered, so '--variant debug' means default+debug; "
                             "components offering neither default nor a requested "
                             "variant are skipped (so '--variant default' skips "
                             "the runtimes). With no --variant, all advertised "
                             "configs are built.")
    parser.add_argument("-C", "--clean", action="store_true",
                        help="prepend an 'install/clean' task that wipes the "
                             "install directory (the versioned symlink target, "
                             "not just the symlink) before building. Per-"
                             "component 'clean' tasks (build dirs) are always "
                             "listed and run like any other task.")
    parser.add_argument("-n", "--dry-run", action="store_true",
                        help="show what would run without executing")
    parser.add_argument("--components", action="store_true",
                        help="print the resolved, ordered component list and exit")
    parser.add_argument("--log-dir", default=None,
                        help="directory for per-task logs "
                             "(default: <BUILD_DIR>/aomp_build_logs)")
    # Environment knobs passed to child build scripts.
    parser.add_argument("-j", "--jobs", type=int, default=None,
                        help="parallel build threads (AOMP_JOB_THREADS)")
    parser.add_argument("--ninja", dest="ninja", action="store_true", default=None,
                        help="use ninja (AOMP_USE_NINJA=1)")
    parser.add_argument("--no-ninja", dest="ninja", action="store_false",
                        help="do not use ninja (AOMP_USE_NINJA=0)")
    parser.add_argument("--ccache", dest="ccache", action="store_true", default=None,
                        help="use ccache (AOMP_USE_CCACHE=1)")
    parser.add_argument("--no-ccache", dest="ccache", action="store_false",
                        help="do not use ccache (AOMP_USE_CCACHE=0)")
    parser.add_argument("--gfx", default=None, metavar="LIST",
                        help="GPU target list (GFXLIST)")
    parser.add_argument("--build-type", action="append", default=[],
                        metavar="SPEC",
                        help="CMake build type (BUILD_TYPE): 'type' (global) or "
                             "'comp=type' (per-component). Comma-separated and/or "
                             "repeatable, e.g. 'project=Debug,comgr=Debug'.")
    parser.add_argument("--sudo", action="store_true",
                        help="install with sudo (SUDO=yes)")
    # Environment isolation.
    parser.add_argument("--inherit-path", action="store_true",
                        help="use the caller's PATH for child build scripts "
                             "instead of the controlled default "
                             f"({DEFAULT_CHILD_PATH})")
    parser.add_argument("--pass-env", action="append", default=[], metavar="VARS",
                        help="leak named environment variable(s) from the caller "
                             "into the otherwise-isolated child environment; "
                             "comma-separated and/or repeatable "
                             "(e.g. --pass-env CC,CXX,LD_LIBRARY_PATH)")
    # Version manifest.
    parser.add_argument("--export-manifest", nargs="?", const="", metavar="FILE",
                        help="export a git fingerprint manifest and exit "
                             "(default path under <BUILD_DIR>/manifests)")
    parser.add_argument("--import-manifest", metavar="FILE",
                        help="check out recorded git SHAs before building")
    return parser


def main(argv: list[str]) -> int:
    args = build_arg_parser().parse_args(argv)

    cfg = parse_cudf(args.config)
    config_name = os.path.splitext(os.path.basename(args.config))[0]
    child_env = build_child_env(args)
    env_info = discover_env(child_env)

    components = resolve_components(cfg, args.add, args.remove)

    if args.components:
        for comp in components:
            print(comp)
        return 0

    build_dir = env_info["BUILD_DIR"]
    log_dir = args.log_dir or os.path.join(build_dir, "aomp_build_logs")
    # Completion stamps live alongside the log dir.
    stamp_dir = os.path.join(os.path.dirname(os.path.abspath(log_dir)), "stamps")
    manifest_dir = os.path.join(build_dir, "manifests")

    # Manifest export is a standalone action.
    if args.export_manifest is not None:
        path = args.export_manifest or os.path.join(
            manifest_dir, f"{config_name}-manifest.json"
        )
        export_manifest(cfg, components, child_env, path, config_name)
        return 0

    # Manifest import runs as a pre-step before any task execution.
    if args.import_manifest:
        rc = import_manifest(cfg, components, child_env, args.import_manifest)
        if rc != 0:
            return rc

    global_variants, per_comp_variants = parse_scoped_specs(args.variant)
    build_type_global, build_type_per_comp = parse_build_type_specs(args.build_type)
    tasks = elaborate_tasks(
        cfg, components, child_env, global_variants, per_comp_variants,
    )
    # -C/--clean prepends a pseudo-task that wipes the install directory so a
    # stale install is removed before anything builds. Per-component build-dir
    # "clean" tasks are always in the list and run like any other task.
    if args.clean:
        tasks.insert(0, make_install_clean_task(env_info))

    # `list` selector: print the numbered task list and exit. A green check
    # marks completed tasks, a red cross marks started-but-unfinished ones.
    if args.selectors and args.selectors[0] == "list":
        width = len(str(len(tasks)))
        for i, task in enumerate(tasks, start=1):
            mark = render_mark(task_state(stamp_dir, task))
            print(f"[{i:0{width}d}] [{mark}] {task.name}")
        return 0

    # Bare `continue`: resume from the first task that is not yet done.
    if args.selectors == ["continue"]:
        resume = next(
            (i for i, t in enumerate(tasks)
             if task_state(stamp_dir, t) != "done"),
            None,
        )
        if resume is None:
            print("aomp_build: all tasks already complete")
            return 0
        indices = list(range(resume, len(tasks)))
    else:
        indices = select_tasks(tasks, args.selectors)
    if not indices:
        print("aomp_build: no tasks selected")
        return 0

    return run_tasks(
        tasks, indices, child_env, log_dir, args.dry_run,
        build_type_global=build_type_global,
        build_type_per_comp=build_type_per_comp,
        log_base=build_dir, stamp_dir=stamp_dir,
    )


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
