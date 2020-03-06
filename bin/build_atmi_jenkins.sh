#!/bin/bash
TEST_ATMI_DIR=${TEST_ATMI_DIR:-$HOME/build_atmi_jenkins}
cd $TEST_ATMI_DIR/build

source envsetup.sh

build_thunk(){
  ./build_thunk.sh -c
  ./build_thunk.sh
}

build_hsa(){
  ./build_hsa.sh -c
  ./build_hsa.sh
}

build_lightning(){
  ./build_lightning.sh -c
  ./build_lightning.sh
}

build_devicelibs(){
  ./build_devicelibs.sh -c
  ./build_devicelibs.sh
}

build_comgr(){
  ./build_comgr.sh -c
  ./build_comgr.sh
}
build_atmi(){
  ./build_atmi.sh -c
  ./build_atmi.sh
}

components="thunk hsa lightning devicelibs comgr atmi"

#Start build from given component (./build_atmi_jenkins.sh continue comgr)
if [ "$1" == "continue" ] ; then
  for COMPONENT in $components ; do
    if [ $COMPONENT == "$2" ] ; then
      found=1
    fi
    if [[ $found -eq 1 ]] ; then
      list+="$COMPONENT "
    fi
  done
  components=$list
  if [[ $found -eq 0 ]] ; then
    echo "$2 was not found in the build list!!!"
  fi
#Select which components to build(./build_atmi_jenkins.sh select thunk hsa)
elif [ "$1" == 'select' ] ; then
  for ARGUMENT in $@ ; do
    if [ $ARGUMENT != "$1" ] ; then
      list+="$ARGUMENT "
    fi
  done
  components=$list
fi

for COMPONENT in $components; do
  build_$COMPONENT
done
