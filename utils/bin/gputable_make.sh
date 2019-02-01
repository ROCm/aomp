#!/bin/bash
#
#  gputable_make.sh : Extract Radeon and Nvidia GPU entries from pci.ids file
#                     and create a formatted gputable.txt file. This file
#                     typically gets installed in $AOMP/bin/gputable.txt.
#                     It is used as a cache to search for GPU identifiers 
#                     obtained from the 'lspci -n -m'  command.  See the
#                     mymcpu shell script for how gputable.txt is searched
#                     with the lspci command.
# 
#  This script runs with no arguments. 
#  This script uses these environment variables with these defaults:
#      PCIID_FILE         $HOME/git/aomp/pciids/pci.ids
#                         /usr/share/misc/pci.ids
#                         /usr/share/hwdata/pci.ids
#      SUPPLEMENTAL_FILE  <thisdir>/gputable_supplemental.txt
#      GPUTABLE_FILE      <thisdir>/gputable.txt
#  where <thisdir> is the same directory where makegputable.sh is found. 
#
#  Written 9/17 by Greg Rodgers
#                   

function getdname(){
   local __DIRN=`dirname "$1"`
   if [ "$__DIRN" = "." ] ; then
      __DIRN=$PWD;
   else
      if [ ${__DIRN:0:1} != "/" ] ; then
         if [ ${__DIRN:0:2} == ".." ] ; then
               __DIRN=`dirname $PWD`/${__DIRN:3}
         else
            if [ ${__DIRN:0:1} = "." ] ; then
               __DIRN=$PWD/${__DIRN:2}
            else
               __DIRN=$PWD/$__DIRN
            fi
         fi
      fi
   fi
   echo $__DIRN
}

function reformat_pciids()
{
local _radeon_entries_active=0
local _nvidia_entries_active=0
local _infile=$1
local _outfile=$2
local _found_rad=0
local _found_nv=0
local _venid=""
local _venword=""
local _numtabs=0
local _outentry=""
origIFS=$IFS
IFS=" "
while read line ; do
   # Ignore comments
   if [ "${line:0:1}" != "#" ] ; then 

      _venid=""
      _venword=""
      _numtabs=0
      _outentry=""
 
      # How many tabs are there in the line
      if [ "${line:0:2}" == "		" ] ; then 
         _numtabs=2
      elif [ "${line:0:1}" == "	" ] ; then 
         _numtabs=1
      else 
         _numtabs=0
         _venid=`echo $line | awk '{print $1}'`
         _venword=`echo $line | awk '{print $2}'`
      fi

      if [ "$_venid" == "1003" ] && [ "$_venword" == "ULSI" ] ; then 
         _radeon_entries_active=0
      fi
      if [ "$_venid" == "10df" ] && [ "$_venword" == "Emulex" ] ; then 
         _nvidia_entries_active=0
      fi

      if [ $_radeon_entries_active == 1 ] || [ $_nvidia_entries_active == 1 ] ; then
         if [ $_numtabs == 1 ] ; then 
            notab=`echo $line | tr -d '\011'`
            majorid=`echo $notab | awk '{print $1}'`
            codename=`echo $notab | awk '{print $2}'` 
            restb=`echo $notab | awk '{$1=$2=$3="" ; print $0}'`
            descript=`echo $notab | awk '{$1="" ; print $0}'`
            _found_rad=0
            _found_nv=0
            # Look at all the words in the description for GPU identifiers 
            # or words that would make it not a GPU that we dont care about
            for dwordspaces in $descript ; do 
               dword=`echo $dwordspaces | tr -d '[:space:]'` 
               if [ "$dword" == "[Radeon"  ] ; then 
                  _found_rad=1
               elif [ "$dword" == "Radeon"  ] ; then 
                  _found_rad=1
               elif [ "$dword" == "Vega"  ] ; then 
                  codename=vega
                  _found_rad=1
               elif [ "$dword" == "Raven"  ] ; then 
                  codename=vega
                  _found_rad=1
               elif [ "$dword" == "Fiji"  ] ; then 
                  codename=fiji
                  _found_rad=1
               elif [ "$dword" == "[GeForce" ] || [ "$dword" == "[Tesla" ] || \
                  [ "$dword" == "[Quadro" ] ; then 
                  _found_nv=1
               elif [ "$dword" == "620M/625M/630M/720M]" ] ; then 
                  _found_nv=0
               elif [ "$dword" == "610M/710M/810M/820M" ] ; then 
                  _found_nv=0
               elif [ "$dword" == "555M/635M]" ] ; then 
                  _found_nv=0
               elif [ "$dword" == "220/315]" ] ; then 
                  _found_nv=0
               elif [ "$dword" == "Controller" ] ; then 
                  _found_nv=0
               else
                  ambigous_word=1
               fi
            done
            if [ $_found_rad == 1  ] ; then 
               _outentry="1002:$majorid 0000 0000 ${codename} :$descript"
            elif [ $_found_nv == 1 ] ; then 
               # For nvidia we derive the codename from the ending description 
               # by removing spaces & ] 
              nvcodename=""
              restb=`echo $restb | tr -d ']' `
              # remove words we do not want in the nvidia codename
              for dword in $restb; do 
                 if [ "$dword" != "Mobile" ] &&
                    [ "$dword" != "mobile" ] &&
                    [ "$dword" != "Max-Q" ]  &&
                    [ "$dword" != "12GB" ]  && 
                    [ "$dword" != "16GB" ]  && 
                    [ "$dword" != "5GB" ]  && 
                    [ "$dword" != "6GB" ]  && 
                    [ "$dword" != "24GB" ]  && 
                    [ "$dword" != "32GB" ]  && 
                    [ "$dword" != "Engineering" ]  && 
                    [ "$dword" != "Sample" ]  && 
                    [ "$dword" != "TITAN" ]  && 
                    [ "$dword" != "X" ]  && 
                    [ "$dword" != "Ti" ]  && 
                    [ "$dword" != "448" ]  && 
                    [ "$dword" != "Cores" ]  && 
                    [ "$dword" != "Boost" ]  && 
                    [ "$dword" != "Memory" ]  && 
                    [ "$dword" != "Controller" ]  && 
                    [ "$dword" != "PCI" ]  && 
                    [ "$dword" != "PCIe" ]  && 
                    [ "$dword" != "SXM2" ]  && 
                    [ "$dword" != "Express" ]  && 
                    [ "$dword" != "Bridge" ]  && 
                    [ "$dword" != "Co-Processor" ]  && 
                    [ "$dword" != "SMBus" ]  && 
                    [ "$dword" != "OHCI" ]  && 
                    [ "$dword" != "USB" ]  && 
                    [ "$dword" != "1.1" ]  && 
                    [ "$dword" != "2.0" ]  && 
                    [ "$dword" != "Mac" ]  && 
                    [ "$dword" != "Edition" ]  && 
                    [ "$dword" != "Rev." ]  && 
                    [ "$dword" != "2" ]  && 
                    [ "$dword" != "LPC" ]  && 
                    [ "$dword" != "OEM" ]  && 
                    [ "$dword" != "SE" ]  && 
                    [ "$dword" != "v2" ]  && 
                    [ "$dword" != "/" ]  && 
                    [ "$dword" != "Quadro" ]  && 
                    [ "$dword" != "M500M" ]  && 
                    [ "$dword" != "DGXS" ]  && 
                    [ "$dword" != "Ultra" ]  && 
                    [ "$dword" != "3GB" ]  ; then 
                    nvcodename="$nvcodename $dword"
                 fi
              done
              #  now remove all spaces
              nvcodename=`echo $nvcodename | tr -d '[:space:]' `
               _outentry="10de:$majorid 0000 0000 ${nvcodename} :$descript"
            else
               _outentry=""
            fi
         elif [ $_numtabs == 2 ] ; then 
            # Only look at these 2-tab entries if the previous 1-tab entry
            # had a GPU that we were looking for. 
            if [ $_found_rad == 1 ] || [ $_found_nv == 1 ] ; then 
               notab=`echo $line | tr -d '\011'`
               code1=`echo $notab | awk '{print $1}'`
               code2=`echo $notab | awk '{print $2}'`
               rest=`echo $notab | awk '{$1=$2=$3="" ; print $0}'`
               descript=`echo $notab | awk '{$1=$2="" ; print $0}'`
               # Get the codename from previous 1 tab entry
               if [ $_found_rad == 1 ] ; then 
                  _outentry="1002:$majorid $code1 $code2 ${codename} :$descript"
               elif [ $_found_nv == 1 ] ; then 
                  _outentry="10de:$majorid $code1 $code2 ${nvcodename} :$descript"
               else
                  _outentry=""
               fi
            fi
         else
            _outentry=""
            echo "ERROR line found with no tabs found while gpu read active"
            exit 1 
         fi
         if [ "$_outentry" != "" ] ; then
            echo $_outentry >> "$_outfile"
         fi
      fi # end of radeon or nvidia entries active

      if [ "$_venid" == "1002" ] && [ "$_venword" == "Advanced" ] ; then 
         _radeon_entries_active=1
      fi
      if [ "$_venid" == "10de" ] && [ "$_venword" == "NVIDIA" ] ; then 
         _nvidia_entries_active=1
      fi
   fi
done < "$_infile"
IFS=$origIFS
}

# -------------------- Main shell starts here -----------------------

thisdir=$(getdname $0)
# pick up output file names from env vars GPUTABLE_FILE & SUPPLEMENTAL_FILE
GPUTABLE_FILE=${GPUTABLE_FILE:-$thisdir/gputable.txt}
SUPPLEMENTAL_FILE="$thisdir/gputable_supplemental.txt"

echo 
# Look for pci.ids file in several locations 
if [ -z $PCIID_FILE ] ; then 
   # Best place to get most up-to-date pci.ids is github
   if [ -f $HOME/git/aomp/pciids/pci.ids ] ; then 
      echo " WARNING:  Getting pci.ids file from $HOME/git/aomp/pciids/pci.ids"
      PCIID_FILE="$HOME/git/aomp/pciids/pci.ids"
   elif [ -f /usr/share/misc/pci.ids ] ; then 
      echo " WARNING:  Getting pci.ids file from /usr/share/misc/pci.ids"
      PCIID_FILE="/usr/share/misc/pci.ids"
   elif [ -f /usr/share/hwdata/pci.ids ] ; then 
      echo " WARNING:  Getting pci.ids file from /usr/share/hwdata/pci.ids"
      PCIID_FILE="/usr/share/hwdata/pci.ids"
   else
      echo " ERROR:  Could not find pci.ids file"
      exit 1
   fi
else 
   if [ ! -f $PCIID_FILE ] ;  then 
      echo " ERROR:  Could not find pci.ids file $PCIID_FILE"
      exit 1
   fi
fi

if [ -f $GPUTABLE_FILE ] ; then 
   echo " WARNING:  Existing file $GPUTABLE_FILE will be overwritten."
fi
TMPFILE="/tmp/gputable$$"
TMPFILE2="/tmp/gputable2$$"
touch $TMPFILE
echo  
echo " Calling reformat_pciids ... "
echo "    Input:        $PCIID_FILE"
echo "    Temp output:  $TMPFILE"
reformat_pciids $PCIID_FILE $TMPFILE
if [ -f $SUPPLEMENTAL_FILE ] ; then 
  echo " Adding supplemental file $SUPPLEMENTAL_FILE"
  /bin/cat $SUPPLEMENTAL_FILE >> $TMPFILE
fi
echo " Converting codenames to lowercase ..."
touch $TMPFILE2
while read line ; do
   front=`echo $line | cut -d" " -f1-3`
   codename=`echo $line | cut -d" " -f4 | awk '{ print tolower($0) }' `
   end=`echo $line | cut -d" " -f5-`
   echo "$front $codename $end" >> $TMPFILE2
done < $TMPFILE

echo " Removing duplicate entries ... "
/bin/cat "$TMPFILE2" | sort -u  > "$GPUTABLE_FILE"
echo "    Final output: $GPUTABLE_FILE"
echo " Done "
echo
rm $TMPFILE
rm $TMPFILE2

exit 0
