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
AOMP_VERSION_STRING=${AOMP_VERSION_STRING:-19.0-3}
AOMP_VERSION=${AOMP_VERSION:-19.0}
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
    elif [[ "$line" =~ "ubuntu:22" ]]; then
      url_array["ubuntu2204"]=$line
    elif [[ "$line" =~ "centos:7" ]]; then
      url_array["centos7"]=$line
    elif [[ "$line" =~ "centos8" ]]; then
      url_array["centos8"]=$line
    elif [[ "$line" =~ "centos-9" ]]; then
      url_array["centos9"]=$line
    elif [[ "$line" =~ "suse" ]]; then
      url_array["sles15"]=$line
    fi
  done < $DOCKERX_HOST/docker-urls.txt
else
  echo "Error: $DOCKERX_HOST/docker-urls.txt not found, exiting."
  exit 1
fi

pip_install="python3 -m pip install CppHeaderParser argparse wheel lit lxml barectf pandas"
pip_install_centos7="python3.8 -m pip install CppHeaderParser argparse wheel lit lxml barectf pandas"
# 22.04 workaround for cython/PyYAML bug.
pip_install_2204="python3 -m pip install --ignore-installed --no-cache-dir barectf==3.1.2 PyYAML==5.3.1; python3 -m pip install CppHeaderParser argparse wheel lit lxml pandas"

# Populate prereq arrays
prereq_array["ubuntu1804"]="apt-get -y update && apt-get install -y git cmake wget vim openssl libssl-dev libelf-dev kmod pciutils gcc g++ pkg-config libpci-dev libnuma-dev libffi-dev git python libopenmpi-dev gawk mesa-common-dev libtool python3 texinfo libbison-dev bison flex libbabeltrace-dev python3-pip libncurses5-dev liblzma-dev python3-setuptools python3-dev libpython3.8-dev libudev-dev libgmp-dev debianutils devscripts cli-common-dev rsync sudo && $pip_install"

prereq_array["ubuntu2004"]="apt-get -y update && apt-get install -y git cmake wget vim openssl libssl-dev libelf-dev kmod pciutils gcc g++ pkg-config libpci-dev libnuma-dev libffi-dev git python libopenmpi-dev gawk mesa-common-dev libtool python3 texinfo libbison-dev bison flex libbabeltrace-dev python3-pip libncurses5-dev liblzma-dev python3-setuptools python3-dev libpython3.8-dev libudev-dev libgmp-dev debianutils devscripts cli-common-dev rsync libsystemd-dev libdw-dev libgtest-dev sudo ccache libgmp-dev libmpfr-dev && $pip_install"

prereq_array["ubuntu2204"]="apt-get -y update && apt-get install -y git cmake wget vim openssl libssl-dev libelf-dev kmod pciutils gcc g++ pkg-config libpci-dev libnuma-dev libffi-dev git libopenmpi-dev gawk mesa-common-dev libtool python3 texinfo libbison-dev bison flex libbabeltrace-dev python3-pip libncurses5-dev liblzma-dev python3-setuptools python3-dev libpython3.10-dev libudev-dev libgmp-dev debianutils devscripts cli-common-dev rsync libsystemd-dev libdw-dev libgtest-dev libstdc++-12-dev sudo python3-lxml ccache libgmp-dev libmpfr-dev && $pip_install_2204"

prereq_array["centos7"]="yum install -y make gcc-c++ git cmake wget vim openssl-devel elfutils-libelf-devel pciutils-devel numactl-devel libffi-devel mesa-libGL-devel libtool texinfo bison flex ncurses-devel expat-devel xz-devel libbabeltrace-devel gmp-devel rpm-build rsync systemd-devel gtest-devel libpciaccess-devel elfutils-devel ccache libxml2-devel xz-lzma-compat devtoolset-9 devtoolset-9-libatomic-devel devtoolset-9-elfutils-libelf-devel scl-utils mpfr-devel && yum remove -y python3*"

prereq_array["centos8"]="yum install -y dnf-plugins-core && yum config-manager --set-enabled PowerTools && yum install -y gcc-c++ git cmake wget vim openssl-devel elfutils-libelf-devel pciutils-devel numactl-devel libffi-devel mesa-libGL-devel libtool texinfo bison flex ncurses-devel expat-devel xz-devel libbabeltrace-devel gmp-devel rpm-build rsync systemd-devel gtest-devel elfutils-devel ccache python38 python38-devel mpfr-devel && yum remove -y python36* && $pip_install"

prereq_array["centos9"]="yum install -y dnf-plugins-core gcc-c++ git cmake wget vim openssl-devel elfutils-libelf-devel pciutils-devel numactl-devel libffi-devel mesa-libGL-devel libtool texinfo bison flex ncurses-devel expat-devel xz-devel libbabeltrace-devel gmp-devel rpm-build rsync systemd-devel gtest-devel ccache mpfr-devel && $pip_install"

prereq_array["sles15"]="zypper install -y which cmake wget vim libopenssl-devel elfutils libelf-devel git pciutils-devel libffi-devel gcc gcc-c++ libnuma-devel openmpi2-devel Mesa-libGL-devel libquadmath0 libtool texinfo bison flex babeltrace-devel python3 python3-pip python3-devel python3-setuptools makeinfo libexpat-devel xz-devel gmp-devel rpm-build rsync libdrm-devel libX11-devel systemd-devel libdw-devel hwdata unzip ccache mpfr-devel; $pip_install"

# Some prep
default_os="ubuntu2004 ubuntu2204 centos7 centos8 centos9 sles15"
OS=${OS:-$default_os}
export DOCKER_HOME=/home/release; export DOCKER_AOMP=/usr/lib/aomp; export DOCKER_AOMP_REPOS=/home/release/git/aomp$AOMP_VERSION
exports="export HOME=/home/release; export AOMP=/usr/lib/aomp; export AOMP_REPOS=/home/release/git/aomp$AOMP_VERSION; export AOMP_EXTERNAL_MANIFEST=1; export AOMP_JOB_THREADS=128; export AOMP_SKIP_FLANG_NEW=1"

function getcontainer(){
  echo docker ps -aqf "name=$docker_name"
  container=$(docker ps -aqf "name=$docker_name")
  echo $container
}

function setup(){
  if [ "$system" == "centos7" ]; then
    exports="$exports; export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH"
  fi

  # Pull docker and start
  docker pull ${url_array[$system]}
  docker run -d -it --name="$docker_name" --network=host --privileged --group-add video --cap-add=SYS_PTRACE --device=/dev/kfd --device=/dev/dri --security-opt seccomp=unconfined --ipc=host -v $DOCKERX_HOST:$DOCKERX ${url_array[$system]}

  if [ "$system" == "centos7" ]; then
    # Support for centos7 has reached EOL. Many of the repos no longer use the mirror list url and need switched to baseurl with vault url.
    docker exec -i $docker_name /bin/bash -c "sed -i 's/mirrorlist=/#mirrorlist=/g' /etc/yum.repos.d/CentOS-*.repo; sed -i 's/#\s*baseurl=/baseurl=/g' /etc/yum.repos.d/CentOS-*.repo; sed -i 's/mirror\./vault\./g' /etc/yum.repos.d/CentOS-*.repo"
    docker exec -i $docker_name /bin/bash -c "yum install -y epel-release centos-release-scl"
    docker exec -i $docker_name /bin/bash -c "sed -i 's/mirrorlist=/#mirrorlist=/g' /etc/yum.repos.d/CentOS-*.repo; sed -i 's/#\s*baseurl=/baseurl=/g' /etc/yum.repos.d/CentOS-*.repo; sed -i 's/mirror\./vault\./g' /etc/yum.repos.d/CentOS-*.repo"
  fi

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
    docker exec -i $docker_name /bin/bash -c "zypper refresh"
    docker exec -i $docker_name /bin/bash -c "$exports; ${prereq_array[$system]} 2>&1 | tee $DOCKER_HOME/logs/$system-preq.out"
    set -e
    docker exec -i $docker_name /bin/bash -c "zypper install -y --force libncurses6=6.1-150000.5.15.1; zypper install -y ncurses-devel"
    docker exec -i $docker_name /bin/bash -c "mkdir /tmp/googletest; cd /tmp/googletest; git clone https://github.com/google/googletest; cd googletest; git checkout release-1.14.0; mkdir build; cd build; cmake -DCMAKE_POSITION_INDEPENDENT_CODE=ON ..; make -j16; make -j16 install"
  else
    docker exec -i $docker_name /bin/bash -c "$exports; DEBIAN_FRONTEND=noninteractive ${prereq_array[$system]} 2>&1 | tee $DOCKER_HOME/logs/$system-preq.out"
  fi
  if [ "$system" == "centos7" ]; then
    exports="$exports; source /opt/rh/devtoolset-9/enable"
    docker exec -i $docker_name /bin/bash -c "$exports; cd /home/release; wget https://www.python.org/ftp/python/3.8.13/Python-3.8.13.tgz; tar xf Python-3.8.13.tgz; cd Python-3.8.13; ./configure --enable-optimizations --enable-shared; make altinstall; ln -s /usr/local/bin/python3.8 /usr/bin/python3; $pip_install_centos7"
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
      echo "OS options: ubuntu1804, ubuntu2004, ubuntu2204, centos7, centos8, centos9, sles15"
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
