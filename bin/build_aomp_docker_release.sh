#!/bin/bash
###########################################################
#       aomp_build_docker.sh
# Script to build AOMP releases in various dockers
# Expects a changelog.txt patch file to be in DOCKER_HOST.
# Expects a docker-urls.txt file in DOCKER_HOST to store docker urls.
# Does not use sudo for docker commands.
#
###########################################################

set -e
AOMP_VERSION_STRING=${AOMP_VERSION_STRING:-15.0-0}
AOMP_VERSION=${AOMP_VERSION:-15.0}
#DOCKERX_HOST=${DOCKERX_HOST:-$HOME/dockerx}
DOCKERX_HOST=$HOME/dockerx
#DOCKERX=${DOCKERX:-/dockerx}
DOCKERX=/dockerx
PATCHLOC=${PATCHLOC:-$DOCKERX/changelog.patch}
host_packages=$HOME/aomp-docker-release/$AOMP_VERSION_STRING/packages

#mkdir -p $docker_home; mkdir -p $docker_home/$AOMP_VERSION_STRING/packages
mkdir -p $host_packages

declare -A url_array
declare -A prereq_array

# Populate url arrays with dockers

if [ -f $DOCKERX_HOST/docker-urls.txt ]; then
  while read -r line; do
    if [[ "$line" =~ "ubuntu-base" ]]; then
      url_array["ubuntu1804"]=$line
    elif [[ "$line" =~ "ubuntu20" ]]; then
      url_array["ubuntu2004"]=$line
    elif [[ "$line" =~ "centos7" ]]; then
      url_array["centos7"]=$line
    elif [[ "$line" =~ "centos8" ]]; then
      url_array["centos8"]=$line
    elif [[ "$line" =~ "sles15" ]]; then
      url_array["sles15"]=$line
    fi
  done < $DOCKERX_HOST/docker-urls.txt
else
  echo "Error: $DOCKERX_HOST/docker-urls.txt not found, exiting."
  exit 1
fi

pip_install="python3 -m pip install CppHeaderParser argparse wheel lit"

# Populate prereq arrays
prereq_array["ubuntu1804"]="apt-get -y update && apt-get install -y git cmake wget vim openssl libssl-dev libelf-dev kmod pciutils gcc g++ pkg-config libpci-dev libnuma-dev libffi-dev git python libopenmpi-dev gawk mesa-common-dev libtool python3 texinfo libbison-dev bison flex libbabeltrace-dev python3-pip libncurses5-dev liblzma-dev python3-setuptools python3-dev libpython3.8-dev libudev-dev libgmp-dev debianutils devscripts cli-common-dev rsync sudo && $pip_install"

prereq_array["ubuntu2004"]="apt-get -y update && apt-get install -y git cmake wget vim openssl libssl-dev libelf-dev kmod pciutils gcc g++ pkg-config libpci-dev libnuma-dev libffi-dev git python libopenmpi-dev gawk mesa-common-dev libtool python3 texinfo libbison-dev bison flex libbabeltrace-dev python3-pip libncurses5-dev liblzma-dev python3-setuptools python3-dev libpython3.8-dev libudev-dev libgmp-dev debianutils devscripts cli-common-dev rsync sudo && $pip_install"

prereq_array["centos7"]="yum install -y gcc-c++ git cmake wget vim openssl-devel elfutils-libelf-devel pciutils-devel numactl-devel libffi-devel mesa-libGL-devel libtool texinfo bison flex ncurses-devel expat-devel xz-devel libbabeltrace-devel gmp-devel python3 python3-pip python36-devel python36-setuptools rpm-build rsync && $pip_install"

prereq_array["centos8"]="yum install -y dnf-plugins-core && yum config-manager --set-enabled PowerTools && yum install -y gcc-c++ git cmake wget vim openssl-devel elfutils-libelf-devel pciutils-devel numactl-devel libffi-devel mesa-libGL-devel libtool texinfo bison flex ncurses-devel expat-devel xz-devel libbabeltrace-devel gmp-devel rpm-build rsync && $pip_install"

prereq_array["sles15"]="zypper install -y libgmp10-6.2.0-3.1.x86_64 which cmake wget vim libopenssl-devel elfutils libelf-devel git pciutils-devel python-base libffi-devel gcc gcc-c++ libnuma-devel openmpi2-devel Mesa-libGL-devel libquadmath0 libtool texinfo bison flex babeltrace-devel python3 python3-pip python3-devel python3-setuptools makeinfo ncurses-devel libexpat-devel xz-devel gmp-devel rpm-build rsync; $pip_install"

# Some prep
default_os="ubuntu1804 ubuntu2004 centos7 centos8 sles15"
OS=${OS:-$default_os}
export DOCKER_HOME=/home/release; export DOCKER_AOMP=/usr/lib/aomp; export DOCKER_AOMP_REPOS=/home/release/git/aomp$AOMP_VERSION
exports="HOME=/home/release; export AOMP=/usr/lib/aomp; export AOMP_REPOS=/home/release/git/aomp$AOMP_VERSION; export AOMP_EXTERNAL_MANIFEST=1"

function getcontainer(){
  echo docker ps -aqf "name=$docker_name"
  container=$(docker ps -aqf "name=$docker_name")
  echo $container
}

function setup(){
  if [ "$system" == "centos7" ]; then
    exports="$exports; source /opt/rh/devtoolset-7/enable"
  fi

  # Pull docker and start
  docker pull ${url_array[$system]}
  docker run -d -it --name="$docker_name" --network=host --privileged --group-add video --cap-add=SYS_PTRACE --device=/dev/kfd --device=/dev/dri --security-opt seccomp=unconfined --ipc=host -v $DOCKERX_HOST:$DOCKERX ${url_array[$system]}

  # Change repos for Centos 8 to enable yum functionality again as it has been vaulted.
  if [ "$system" == "centos8" ]; then
    docker exec -i $docker_name /bin/bash -c "cd /etc/yum.repos.d/; sed -i 's/mirrorlist/#mirrorlist/g' /etc/yum.repos.d/CentOS-*; sed -i 's|#baseurl=http://mirror.centos.org|baseurl=http://vault.centos.org|g' /etc/yum.repos.d/CentOS-*"
  elif [ "$system" == "sles15" ]; then
    # Create symbolic link for libquadmath and rename /usr/src/packages as that prevents rpmbuild from getting the correct source directory.
    docker exec -i $docker_name /bin/bash -c "ln -s /usr/lib64/libquadmath.so.0 /usr/lib64/libquadmath.so"
    docker exec -i $docker_name /bin/bash -c "mv /usr/src/packages /usr/src/packages-temp"
  fi

  # Setup directory structure
  docker exec -i $docker_name /bin/bash -c "$exports; mkdir -p $DOCKER_AOMP_REPOS; mkdir -p $DOCKER_HOME/logs"

  # Hardcode timezone for tzdata install to avoid an interactive prompt
  docker exec -i $docker_name /bin/bash -c "$exports; ln -fs /usr/share/zoneinfo/America/Chicago /etc/localtime"

  # Install prerequisite system packages
  if [ "$system" == "sles15" ]; then
    set +e
    docker exec -i $docker_name /bin/bash -c "$exports; DEBIAN_FRONTEND=noninteractive ${prereq_array[$system]} 2>&1 | tee $DOCKER_HOME/logs/$system-preq.out"
    set -e
  else
    docker exec -i $docker_name /bin/bash -c "$exports; DEBIAN_FRONTEND=noninteractive ${prereq_array[$system]} 2>&1 | tee $DOCKER_HOME/logs/$system-preq.out"
  fi
  # Run build_prerequisites.sh to build cmake, hwloc, rocmsmi, etc
  docker exec -i $docker_name /bin/bash -c "$exports; cd $DOCKER_AOMP_REPOS; git clone -b aomp-$AOMP_VERSION_STRING https://github.com/ROCm-Developer-Tools/aomp; cd aomp/bin; ./build_prereq.sh 2>&1 | tee $DOCKER_HOME/logs/$system-prereq.out"

  # Clone repos
  docker exec -i $docker_name /bin/bash -c "$exports; cd $DOCKER_AOMP_REPOS/aomp/bin; ./clone_aomp.sh 2>&1 | tee $DOCKER_HOME/logs/$system-clone.out"
}

function build(){
  docker exec -i $docker_name /bin/bash -c "$exports; cd $DOCKER_AOMP_REPOS/aomp/bin; ./build_aomp.sh 2>&1 | tee $DOCKER_HOME/logs/$system-build.out"
}

function package(){
  getcontainer
  docker exec -i $docker_name /bin/bash -c "grep 'END build_aomp' $DOCKER_HOME/logs/$system-build.out"
  if [ "$?" -eq 0 ]; then
    if [[ "$system" =~ "ubuntu" ]]; then
      # Update changelog with user patch
      docker exec -i $docker_name /bin/bash -c "$exports; cd $DOCKER_AOMP_REPOS/aomp/bin; patch debian/changelog /dockerx/changelog.patch"
      # Build debian
      docker exec -i $docker_name /bin/bash -c "$exports; cd $DOCKER_AOMP_REPOS/aomp/bin; ./build_fixups.sh; DOCKER=1 ./build-deb-aomp.sh 2>&1 | tee $DOCKER_HOME/logs/$system-package.out"
      # Copy to host
      docker cp $container:/tmp/build-deb/debs/. $host_packages
    else
      # Update changelog with user patch and change aomp version header
      docker exec -i $docker_name /bin/bash -c "$exports; cd $DOCKER_AOMP_REPOS/aomp/bin; patch debian/changelog /dockerx/changelog.patch; sed -i -e 's/aomp (.*)/aomp ($AOMP_VERSION_STRING)/g' debian/changelog"
      # Build rpm
      if [ "$system" == "centos7" ]; then
        docker exec -i $docker_name /bin/bash -c "$exports; cd $DOCKER_AOMP_REPOS/aomp/bin; ./build_fixups.sh; DOCKER=1 ./build-rpm.sh aomp_CENTOS_7 2>&1 | tee $DOCKER_HOME/logs/$system-package.out"
      else
        docker exec -i $docker_name /bin/bash -c "$exports; cd $DOCKER_AOMP_REPOS/aomp/bin; ./build_fixups.sh; DOCKER=1 ./build-rpm.sh 2>&1 | tee $DOCKER_HOME/logs/$system-package.out"
      fi
      # Copy to host
      docker cp $container:/tmp/home/rpmbuild/RPMS/x86_64/. $host_packages
    fi
  fi
}

if [ "$#" -eq 0 ]; then
  target="setup build package"
fi

while [ "$1" != "" ];
do
  case $1 in
    -s | --setup | setup)
      target="setup" ;;
    -b | --build | build)
      target="build" ;;
    -p | --package | package)
      target="package" ;;
    -h | --help | help)
      echo "------------------------ Help ---------------------------------"
      echo "Script to build AOMP releases in various dockers."
      echo "Expects a changelog.txt patch file to be in DOCKER_HOST."
      echo "Expects a docker-urls.txt file in DOCKER_HOST to store docker urls."
      echo "Does not use sudo for docker commands."
      echo ""
      echo "OS=<operating system/s> ./build_aomp_docker.sh [-option]"
      echo ""
      echo "OS options: ubuntu1804, ubuntu2004, centos7, centos8, sles15"
      echo "  default:  all"
      echo
      echo "options(accepts one at a time): -s (setup), -b (build), -p (package), -h (help)"
      echo "  default: -s, -b, -p"
      echo
      echo "example-1: ./build_aomp_docker.sh"
      echo "example-2: OS=\"ubuntu1804 centos7\" ./build_aomp_docker.sh"
      echo "example-3: OS=sles15 ./build_aomp_docker.sh -s"
      echo "---------------------------------------------------------------"
      exit ;;
    *)
      echo $1 option not recognized ; exit 1 ;;
  esac
  shift 1
done

# Begin
for system in $OS; do
  # Verify operating system is supported
  if [ "${prereq_array[$system]}" == "" ]; then
    echo $system is not a supported os. Choose from: $default_os.
    exit 1
  fi
  echo "Building AOMP in $system docker."

  docker_name="$system-$AOMP_VERSION_STRING"
  # Setup/Build/Package
  for step in $target; do
    echo Executing Step: $step
    $step
  done
done
