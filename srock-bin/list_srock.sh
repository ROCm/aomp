#!/bin/bash
#
#Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
#SPDX-License-Identifier:  MIT
#
#  list_srock.sh: List the version of TheRock and submodules
#
# --- Start standard header to set SROCK environment variables ----
realpath=$(realpath "$0")
thisdir=$(dirname "$realpath")
. "$thisdir/srock_common_vars"
# --- end standard header ----

cd "$SROCK_THEROCK_DIR" || exit

do_sub_status=0
show_fmt=1
show_unfmt=0
while [ $# -gt 0 ]; do		# check arguments
   case "$1" in
      -u)
         show_fmt=0
         show_unfmt=1
	 ;;
      -v)
         do_sub_status=1
	 ;;
      -q|-t)
         do_sub_status=0
	 ;;
      *)
	 echo "Usage: srock-list [OPTION]" >& 2
	 echo "    -q  quick   (exclude 'submodule status')" >& 2
	 echo "    -v  verbose (include 'submodule status')" >& 2
	 echo "    -u  show as unformatted CSV" >& 2
	 exit 1 ;;
   esac
   shift
done
declare -a tr_dtop
declare -a tr_dsub
# shellcheck disable=SC2116 # echo intended
# shellcheck disable=SC2046 # word splitting in subcommands is acceptable
tr_dtop=( "$(echo $(git branch --show-current)\|TheRock\|TheRock\|parent\|$(git log -1 --format="%H|%as|%cn|%an"))" )
# shellcheck disable=SC2016 # single quotes intended
# shellcheck disable=SC2207 # intended split on newline
IFS=$'\n' tr_dsub=( $(git submodule -q foreach 'echo $(git branch --show-current)\|$sm_path\|$name\|$sha1\|$(git log -1 --format="%H|%as|%cn|%an")') )
# shellcheck disable=SC2207 # intended split on newline
declare -a tr_data
tr_data=( "${tr_dtop[@]}" "${tr_dsub[@]}" )

declare -a tr_fields
declare -a tr_unders
tr_fields=("branch" "path" "repo name" "sub SHA" "head SHA" "updated" "commitor" "for author")
tr_unders=("------" "----" "---------" "-------" "--------" "-------" "--------" "----------")
if [[ $do_sub_status -ne 0 ]]; then
    tr_fields+=("sub SHA tag")
    tr_unders+=("-----------")
fi
declare -A tr_htags

# collect SHA tags from submodule status
if [[ $do_sub_status -ne 0 ]]; then
    declare -a tr_stat
    # shellcheck disable=SC2207 # intended split on newline
    IFS=$'\n' tr_stat=( $(git submodule status) )
    for entry in "${tr_stat[@]}"; do
        IFS=" " read -r -a en_split <<< "$entry"
        sub_sha=${en_split[0]}
        sub_tag=${en_split[2]}
        # echo "$entry: $sub_sha -> $sub_tag"
        tr_htags[$sub_sha]=$sub_tag
    done
fi

# formatted output
if [[ $show_fmt -ne 0 ]]; then
    echo "TheRock submodules:"
    fmt="%-20.20s %-44.44s %-21.21s %-10.10s %-10.10s %-10.10s %-10.10s %-19.19s %s\n"
    # shellcheck disable=SC2059 # variable format intended
    printf "$fmt" "${tr_fields[@]}"
    # shellcheck disable=SC2059 # variable format intended
    printf "$fmt" "${tr_unders[@]}"
    for entry in "${tr_data[@]}"; do
        IFS='|' read -r -a en_split <<< "$entry"
        sub_sha=${en_split[3]}
        head_sha=${en_split[4]}
        if [ "$sub_sha" == "$head_sha" ]; then
            en_split[4]="same"
        fi
        tag=""
        if [[ -v tr_htags[$sub_sha] ]]; then
            tag=${tr_htags[$sub_sha]}
        fi
        # shellcheck disable=SC2059 # variable format intended
        printf "$fmt"  "${en_split[@]}" "$tag"
    done
fi

# unformatted output
if [[ $show_unfmt -ne 0 ]]; then
    echo "TheRock submodules versions (unformatted):"
    fmt="%s|%s|%s|%s|%s|%s|%s|%s|%s|%s\n"
    # shellcheck disable=SC2059 # variable format intended
    printf "$fmt" "${tr_fields[@]}"
    for entry in "${tr_data[@]}"; do
        IFS='|' read -r -a en_split <<< "$entry"
        sub_sha=${en_split[3]}
        tag=${tr_htags[$sub_sha]}
        echo "$entry|$tag"
    done
fi
