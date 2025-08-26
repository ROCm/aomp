#!/bin/bash
#
# tr_add_info.sh: Update the AOMP release info file
#
#
# --- Start standard header to set AOMP environment variables ----
realpath=$(realpath "$0")
thisdir=$(dirname "$realpath")
. "$thisdir/tr_aomp_common_vars"
# --- end standard header ----

_tmpfile=/tmp/info_file_tmp$$.info
AOMP_INFO_FILE=$TR_AOMP_REPOS/aomp/tr_aomp/aomp_${AOMP_VERSION_STRING}.info
_field_name=$1
shift
_field_values=$@
if [ -z $_field_name ] || [ -z "$_field_values" ] ; then 
   echo "ERROR: $0 requires a field name and values"
   exit 1
fi

# Remove existing field name if it exists
grep -v "^${_field_name}:" $AOMP_INFO_FILE > $_tmpfile
cp $_tmpfile $AOMP_INFO_FILE

echo ${_field_name}: $_field_values >> $AOMP_INFO_FILE

rm $_tmpfile

echo "------- DUMP OF $AOMP_INFO_FILE ----------"
cat $AOMP_INFO_FILE
