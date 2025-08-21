#!/bin/bash
#
#  tr_list_aomp_branches.sh: list key information about all the repos found
#                            in the tr_aompi_$AOMP_VERSION.xml file
 
# --- Start standard header to set AOMP environment variables ----
realpath=$(realpath "$0")
thisdir=$(dirname "$realpath")
. "$thisdir/tr_aomp_common_vars"
# --- end standard header ----
 
_therockdir=$TR_AOMP_REPOS/TheRock
_curdir=$PWD

_manifest_file=$TR_AOMP_REPOS/aomp/tr_aomp/tr_aomp.xml

HEADERS=0

   if [ ! -f "$_manifest_file" ] ; then
      echo "ERROR manifest file missing: $_manifest_file"
      exit 1
   fi
   tmpfile=/tmp/mlines$$
   # HACK: The manifest file must be one project line per repo
   grep project < "$_manifest_file" > "$tmpfile"
   if [ $HEADERS == 1 ] ; then 
      printf "MANIFEST FILE: %40s\n" "$_manifest_file"
      printf "%20s %20s %30s %30s \n" "remote" "branch" "path" "repo name"
      printf "%20s %20s %30s %30s \n" "------" "------" "----" "---------"
   fi
   while read -r line; do
      line_is_good=1
      remote=$(echo "$line" | grep remote | cut -d"=" -f2)
      sha_key_used=0
      COSHAKEY=""
      for field in $line; do
         if [[ "$field" =~ remote=\"([^\"]*)\" ]]; then
           remote=${BASH_REMATCH[1]}
         fi
         if [[ "$field" =~ name=\"([^\"]*)\" ]]; then
           name=${BASH_REMATCH[1]}
	   name=${name%*.git}
         fi
         if [[ "$field" =~ path=\"([^\"]*)\" ]]; then
           path=${BASH_REMATCH[1]}
         fi
         if [[ "$field" =~ upstream=\"([^\"]*)\" ]]; then
           COBRANCH=${BASH_REMATCH[1]}
           sha_key_used=1
         fi
         if [[ "$field" =~ revision=\"([^\"]*)\" ]] && [ "$sha_key_used" == 1 ]; then
           COSHAKEY=${BASH_REMATCH[1]}
         elif [[ "$field" =~ revision=\"([^\"]*)\" ]]; then
           COBRANCH=${BASH_REMATCH[1]}
         fi
      done
      if [ "$remote" == "roc" ] ; then
         repo_web_location=$GITROC
      elif [ "$remote" == "gerritgit" ] ; then
         repo_web_location=$GITGERRIT
      elif [ "$remote" == "hwloc" ] ; then
         repo_web_location=$GITHWLOC
      elif [ "$remote" == "githubemu-lightning" ] ; then
         repo_web_location=$GITLIGHTNINGINTERNAL
      else
         line_is_good=0
      fi
      if [ "$line_is_good" == 1 ] ; then
         printf "%20s %20s %30s %30s\n"  "$remote" "$COBRANCH" "$path" "$name"
      fi  # end line_is_good
   done <"$tmpfile"
   rm "$tmpfile"

   cd $_curdir
   exit

