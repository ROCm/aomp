#!/usr/bin/env python3

"""
run.py - Developer helper for the manylinux buildbot docker images.

Quickly set up an environment from one of the provided image directories
(manylinux-build-only, manylinux-hip-tpl, etc) for local debugging / reproduction.

Examples:
  # Download the base Dockerfile + helper scripts
  python run.py manylinux-build-only --pull

  # Build the base image, the target image, and run a container
  python run.py manylinux-build-only --build

  # Remove the container and image
  python run.py manylinux-build-only --clean
"""

import argparse
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

BASE_IMAGE = "localhost/manylinux:base"
BASE_DOCKERFILE = "build_manylinux_x86_64.Dockerfile"
THEROCK_LINK = "https://raw.githubusercontent.com/ROCm/TheRock/main/dockerfiles"
# Update this when onboarding a new buildbot image.
TARGETS = ["manylinux-build-only", "manylinux-hip-tpl"]

# Necessary files.
BASE_FILES = [
    BASE_DOCKERFILE,
    "install_ccache.sh",
    "install_sccache.sh",
    "install_cmake.sh",
    "install_ninja.sh",
    "install_awscli.sh",
    "install_googletest.sh",
    "install_rust.sh",
    "install_patchelf.sh",
    "install_shared_pythons.sh",
]

GPU_RUN_FLAGS = [
    "--device=/dev/kfd",
    "--device=/dev/dri",
    "--group-add", "video",
    "--group-add", "render",
]


# Helper functions.
def log(msg):
    print(f"[run.py] {msg}", flush=True)


def run_cmd(cmd, check=True, capture=False):
    """Echo and run a command, streaming output. Returns CompletedProcess."""
    log("$ " + " ".join(cmd))
    return subprocess.run(cmd, check=check, text=True, capture_output=capture)


def require_docker():
    if shutil.which("docker") is None:
        sys.exit("error: 'docker' not found on PATH.")


def image_exists(tag):
    res = run_cmd(["docker", "images", "-q", tag], check=False, capture=True)
    return bool(res.stdout.strip())


def container_exists(name):
    res = run_cmd(
        ["docker", "ps", "-aq", "-f", f"name=^{name}$"],
        check=False,
        capture=True,
    )
    return bool(res.stdout.strip())


def get_target_dir(target):
    return SCRIPT_DIR / target


def base_context_dir(dest):
    return Path(dest).resolve() / "manylinux-base"


def container_name(args, target):
    return args.name or f"test-{target}"


def download(url, out, retries=3):
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                out.write_bytes(resp.read())
            return
        except Exception as err:
            last_err = err
            log(f"attempt {attempt}/{retries} failed: {err}")
    sys.exit(f"error: failed to download {url}: {last_err}")


# Operations that this script can perform.
def cmd_pull(args):
    ctx = base_context_dir(args.dest)
    ctx.mkdir(parents=True, exist_ok=True)

    log(f"Downloading base build context into {ctx}")
    for name in BASE_FILES:
        out = ctx / name
        url = f"{THEROCK_LINK}/{name}"

        log(f"fetch: {url}")
        download(url, out)

        if name.endswith(".sh"):
            out.chmod(0o755)

    log("Pull complete.")


def cmd_build(args):
    require_docker()
    target_dir = get_target_dir(args.target)
    ctx = base_context_dir(args.dest)

    if not image_exists(BASE_IMAGE):
        dockerfile = ctx / BASE_DOCKERFILE
        if not dockerfile.is_file():
            sys.exit(
                f"error: {BASE_IMAGE} is missing and base files were not found "
                f"in {ctx}.\n       Run a pull first, e.g. "
                f"python run.py {args.target} --pull --dest {args.dest}"
            )
        log(f"Building base image {BASE_IMAGE}")

        run_cmd([
            "docker", "build",
            "-t", BASE_IMAGE,
            "-f", str(dockerfile),
            str(ctx),
        ])
    else:
        log(f"Base image {BASE_IMAGE} already present.")

    image_tag = args.target
    log(f"Building target image {image_tag}")
    run_cmd([
        "docker", "build",
        "-t", image_tag,
        "-f", str(target_dir / "Dockerfile"),
        str(target_dir),
    ])

    name = container_name(args, args.target)
    if container_exists(name):
        log(f"Removing pre-existing container {name}")
        run_cmd(["docker", "rm", "-f", name], check=False)
    run_args = ["docker", "run", "-dit", "--network=host", "--name", name]

    if not args.no_gpu:
        run_args += GPU_RUN_FLAGS
    run_args += [image_tag, "sleep", "infinity"]

    log(f"Starting container {name}")
    run_cmd(run_args)
    log("Build complete.")
    log(f"Open a shell with: docker exec -it {name} bash")


def cmd_clean(args):
    require_docker()
    name = container_name(args, args.target)

    log(f"Removing container {name}")
    run_cmd(["docker", "rm", "-f", name], check=False)

    log(f"Removing image {args.target}")
    run_cmd(["docker", "rmi", args.target], check=False)

    log(f"Removing base image {BASE_IMAGE}")
    run_cmd(["docker", "rmi", BASE_IMAGE], check=False)

    log("Clean complete.")


# CLI
def build_parser():
    parser = argparse.ArgumentParser(
        description="Helper to set up manylinux buildbot docker images.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument("target", choices=TARGETS, help="image type, e.g. manylinux-build-only")
    parser.add_argument("--pull", action="store_true",
                        help="download base Dockerfile + helper scripts")
    parser.add_argument("--build", action="store_true",
                        help="build images and create/run the container")
    parser.add_argument("--clean", action="store_true",
                        help="remove the container and image(s)")
    parser.add_argument("--dest", default=".",
                        help="base build-context location (default: current dir)")
    parser.add_argument("--name", help="container name (default: test-<target>)")
    parser.add_argument("--no-gpu", action="store_true",
                        help="skip GPU device/group run flags (build)")


    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)

    # By default, perform a pull + build.
    if not (args.pull or args.build or args.clean):
        args.pull = True
        args.build = True

    try:
        if args.clean:
            cmd_clean(args)
        if args.pull:
            cmd_pull(args)
        if args.build:
            cmd_build(args)
    except subprocess.CalledProcessError as err:
        sys.exit(f"error: step failed ({err}); aborting.")


if __name__ == "__main__":
    main()
