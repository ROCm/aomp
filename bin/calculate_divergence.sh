#!/bin/bash

################################################################################
# Calculates and reports the divergence between LLVM upstream and ROCm's LLVM 
# fork by comparing git branches and generating diff statistics and patches.
#
# Environment Variables:
#   SKIP_FETCH    - Skip git fetch operations (default: 0)
#   SILENT        - Suppress informational messages (default: 0)
#   SETUP_ONLY    - Only runs setup process, then exits (default: 0)
#   SKIP_SETUP    - Skips directory & git setup steps (default: 0)
#   LLVM_REPO_DIR - Path to LLVM repository (default: ~/git/llvm-project.diff)
#   LLVM_PATH     - Path to analyze (default: "" for entire repo)
#   COMPONENT     - Index into list of paths (default: "" for entire repo)
#   LLVM_BRANCH   - LLVM branch to compare (default: main)
#   ROCm_BRANCH   - ROCm branch to compare (default: amd-staging)
################################################################################

timestamp="$(date +"%Y-%m-%d_%H-%M")"

source_config="llvm"
target_config="ROCm"
remote_configs=("$source_config" "$target_config")

# configs
declare -A components=(
    [clang]="clang"
    [flang]="flang"
    [offload]="offload"
    [openmp]="openmp"
    [driver]="clang/lib/Driver clang/include/clang/Driver clang/test/Driver"
)

declare -A directories=(
    [llvm]=${LLVM_REPO_DIR:=${HOME}/git/llvm-project.diff}
    [path]=${LLVM_PATH:=""}
    [results]="$(pwd)/results"
)

declare -A diff_args=(
    [component]=${COMPONENT:=""}
    [options]="stat patch"
)

declare -A llvm=(
    [url]="https://github.com/llvm/llvm-project.git"
    [remote]="llvm"
    [branch]=${LLVM_BRANCH:=main}
)

declare -A ROCm=(
    [url]="https://github.com/ROCm/llvm-project.git"
    [remote]="ROCm"
    [branch]=${ROCm_BRANCH:=amd-staging}
)

################################################################################
#   Utility functions for consistent, formatted console output throughout the
#   script execution. Provides two levels of messaging: step headers for major 
#   operations and informational messages for detailed progress updates.
################################################################################
print_step() {
    echo -e "$1..."
}

print_info() {
    if [[ "$SILENT" -eq 1 ]]; then
        return 0
    fi

    local is_first_line=1
    local prefix=" * "

    while IFS= read -r line; do
        printf "%s%s\n" "$prefix" "$line"
        # echo -e "${prefix}${line}"
        if [[ "$is_first_line" -eq 1 ]]; then
            prefix="   "
            is_first_line=0
        fi
    done <<< "$1"
}

################################################################################
#   Sets up the LLVM repository directory. Checks if the directory exists and
#   is a valid git repository. If not, clones the LLVM repository from the
#   configured URL.
################################################################################
setup_directory() {
    print_step "Setting up required directories"
    local git_dir="${directories[llvm]}"
    local parent_dir=$(dirname "$git_dir")

    if [[ -d $git_dir ]] && git -C "$git_dir" rev-parse --is-inside-work-tree &>/dev/null ; then
        print_info "found existing directory $git_dir"
        return 0
    fi

    print_info "creating directory $git_dir"
    mkdir -p $parent_dir
    cd $parent_dir
    git clone ${llvm[url]} "$(basename "$git_dir")"
}

################################################################################
#   Sets up git remotes for both LLVM and ROCm repositories. Checks if remotes
#   with the configured URLs already exist. If a remote exists with a different
#   name, updates the configuration to use the existing name. Otherwise, adds
#   the remote with the configured name.
################################################################################
setup_remotes() {
    print_step "Setting up required remotes"

    cd "${directories[llvm]}"

    for name in "${remote_configs[@]}"; do
        local -n config="$name"
        local url="${config[url]}"
        local remote="${config[remote]}"

        match=$(git remote -v | grep -E "\s${url}\s" | awk '{print $1}' | head -n 1)

        if [ -n "$match" ]; then
            print_info "found remote $url ($match)"
            config[remote]=$match
        else
            print_info "remote $url not found, adding as $remote"
            git remote add $remote $url
        fi
    done
}

################################################################################
#   Enriches configuration arrays with formatted reference strings for different
#   use cases. Generates three formats for each config: 'ref' (remote/branch) 
#   for git commands, 'file' (remote_branch) for safe filenames, and 'name' 
#   (remote/branch) for display purposes.
################################################################################
update_configs() {
    print_step "Updating configs"

    local -A formats=(
        [ref]="/"
        [file]="_"
        [name]="/"
    )

    for c in "${remote_configs[@]}"; do
        local -n config="$c"
        local remote="${config[remote]}"
        local branch="${config[branch]}"

        for key in "${!formats[@]}"; do
            config[$key]="$remote${formats[$key]}$branch"
        done
    done
}

################################################################################
#   Validates and formats paths for use in a `git diff` command based on a
#   pre-defined component name or a list of explicit paths. It iterates through
#   the specified paths, verifies that each exists, and formats them
#   appropriately (e.g., adding a trailing slash for directories). The function
#   populates the `diff_args` associative array with a formatted `path` argument
#   string (e.g., `-- path1/ path2`) and a `file_suffix` for use in output
#   filenames. If no valid paths are found, it clears these arguments to ensure
#   the diff runs on the entire repository.
################################################################################
check_paths() {
    local c="${diff_args[component]}"
    local p="${directories[path]}"

    if [[ -z "$c" && -z "$p" ]]; then
        return 0
    fi

    print_step "Checking paths"
    
    diff_args[file_suffix]=$(
        if [[ ! -z "$c" ]]; then
            echo "$c"
        else
            echo "${p%/}"
        fi
    )

    for path in ${components[$c]} $p; do
        local full_path="${directories[llvm]}/$path"
        if [[ ! -e "$full_path" ]]; then
            print_info "$path does not exist, skipping"
            continue
        fi
        diff_args[path]+=$(
            if [[ -d "$full_path" ]]; then
                echo " ${path%/}/"
            elif [[ -f "$full_path" ]]; then
                echo " $path"
            fi
        )
    done

    if [[ -z "${diff_args[path]}" ]]; then
        diff_args[file_suffix]=""
        return 0
    fi

    diff_args[file_suffix]="--${diff_args[file_suffix]}"
    diff_args[path]="-- ${diff_args[path]}"
}

################################################################################
#   Fetches the latest changes from configured git remotes for both LLVM and
#   ROCm repositories. Uses dry-run to check if updates are available before
#   fetching. Skips all fetch operations if SKIP_FETCH is set to "true".
################################################################################
update_sources() {
    if [[ "$SKIP_FETCH" -eq 1 ]]; then
        return 0
    fi

    print_step "Updating sources"

    for config in "${remote_configs[@]}"; do
        local -n c="$config"
        local remote="${c[remote]}"
        local branch="${c[branch]}"
        
        fetch_output=$(git fetch --dry-run "$remote" "$branch" 2>&1)
        if [ -z "$fetch_output" ] ; then
            print_info "$remote/$branch already up to date"
        else
            git fetch $remote $branch
        fi
    done
}

################################################################################
#   Calculates the divergence between source and target branches using git's
#   merge-base comparison. Generates diff output files in multiple formats
#   (stat and patch by default) and saves them to the results directory with
#   timestamped filenames. Optionally limits analysis to a specific paths.
################################################################################
calculate_differences() {
    local -n a="$source_config"
    local -n b="$target_config"
    local a_ref="${a[ref]}"
    local b_ref="${b[ref]}"
    local path_arg="${diff_args[path]}"

    local filename="${a[file]}-${b[file]}${diff_args[file_suffix]}[$timestamp]"
    filename="${filename//\//_}"

    print_step "Calculating difference"
    print_info "based on: $(git show -s $(git merge-base $a_ref $b_ref))"
    print_info "between: ${a[name]}
    and: ${b[name]}"

    if [[ ! -z "$path_arg" ]] ; then
        local cleaned_path_arg="${path_arg#-- }"
        print_info "only including files from:${cleaned_path_arg// /$'\n'}"
    fi

    print_info "$(git diff --merge-base --shortstat $a_ref $b_ref $path_arg)"

    IFS=' ' read -ra operations  <<< "${diff_args[options]}"
    for op in "${operations[@]}"; do
        print_info "calculating git diff --$op"
        git diff --merge-base --$op $a_ref $b_ref $path_arg > "${directories[results]}/$filename.$op"
    done
}

################################################################################
#   Normalizes a predefined list of boolean-like environment variables into a
#   consistent integer format (1 for true, 0 for false). It reads each variable
#   name from the `boolean_vars` array, interprets its value, and overwrites
#   the global variable with either a 1 or a 0.
#
#   This function handles case-insensitive "true" values (e.g., true, 1, yes, y)
#   and "false" values (e.g., false, 0, no, n). If a variable is unset or has
#   an unrecognized value, it defaults to 0 (false).
################################################################################
process_environment() {
    print_step "Process environment"
    local boolean_vars=("SKIP_FETCH" "SILENT" "SETUP_ONLY" "SKIP_SETUP")

    for var_name in "${boolean_vars[@]}"; do
        local -n var="$var_name"
        var=${var:-0}

        case "${var,,}" in
            true|1|yes|y)
                var=1
                ;;
            false|0|no|n)
                var=0
                ;;
            *)
                var=0
                ;;
        esac

        print_info "$var_name=$var"
    done
}

main() {
    local working_dir=$(pwd)
    local results_dir="${directories[results]}"
    mkdir -p $results_dir
    
    local log_file="${directories[results]}/$(basename "${0%.*}")[$timestamp].log"

    exec > >(tee -a "$log_file") 2>&1

    process_environment

    if [[ "$SKIP_SETUP" -eq 0 ]]; then
        setup_directory
        setup_remotes
    fi

    check_paths

    if [[ "$SETUP_ONLY" -eq 1 ]]; then
        return 0
    fi

    update_configs
    update_sources
    calculate_differences

    print_step "Cleaning up"
    print_info "output files written to: $results_dir"
    cd $working_dir
}

main "$@"
