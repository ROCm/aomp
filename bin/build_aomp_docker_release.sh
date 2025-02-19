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
set -x
AOMP_VERSION_STRING=${AOMP_VERSION_STRING:-20.0-2}
AOMP_VERSION=${AOMP_VERSION:-20.0}
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
    elif [[ "$line" =~ "ubuntu:noble" ]]; then
      url_array["ubuntu2404"]=$line
    elif [[ "$line" =~ "centos:7" ]]; then
      url_array["centos7"]=$line
    elif [[ "$line" =~ "centos8" ]]; then
      url_array["centos8"]=$line
    elif [[ "$line" =~ "centos-9" ]]; then
      url_array["centos9"]=$line
    elif [[ "$line" =~ "ubi8" ]]; then
      url_array["rhel8"]=$line
    elif [[ "$line" =~ "ubi9" ]]; then
      url_array["rhel9"]=$line
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

# 24.04 uses python virtual environment
pip_install_2404="python3 -m venv /opt/venv; PATH=/opt/venv/bin:$PATH python3 -m pip install CppHeaderParser argparse lxml pandas setuptools PyYAML pandas"

# Populate prereq arrays
prereq_array["ubuntu1804"]="apt-get -y update && apt-get install -y git cmake wget vim openssl libssl-dev libelf-dev kmod pciutils gcc g++ pkg-config libpci-dev libnuma-dev libffi-dev git python libopenmpi-dev gawk mesa-common-dev libtool python3 texinfo libbison-dev bison flex libbabeltrace-dev python3-pip libncurses5-dev liblzma-dev python3-setuptools python3-dev libpython3.8-dev libudev-dev libgmp-dev debianutils devscripts cli-common-dev rsync sudo && $pip_install"

p2rereq_array["ubuntu2004"]="apt-get -y update && apt-get install -y git cmake wget vim openssl libssl-dev libelf-dev kmod pciutils gcc g++ pkg-config libpci-dev libnuma-dev libffi-dev git python libopenmpi-dev gawk mesa-common-dev libtool python3 texinfo libbison-dev bison flex libbabeltrace-dev python3-pip libncurses5-dev liblzma-dev python3-setuptools python3-dev libpython3.8-dev libudev-dev libgmp-dev debianutils devscripts cli-common-dev rsync libsystemd-dev libdw-dev libgtest-dev sudo ccache libgmp-dev libmpfr-dev && $pip_install"

prereq_array["ubuntu2204"]="apt-get -y update && apt-get install -y git cmake wget vim openssl libssl-dev libelf-dev kmod pciutils gcc g++ pkg-config libpci-dev libnuma-dev libffi-dev git libopenmpi-dev gawk mesa-common-dev libtool python3 texinfo libbison-dev bison flex libbabeltrace-dev python3-pip libncurses5-dev liblzma-dev python3-setuptools python3-dev libpython3.10-dev libudev-dev libgmp-dev debianutils devscripts cli-common-dev rsync libsystemd-dev libdw-dev libgtest-dev libstdc++-12-dev sudo python3-lxml ccache libgmp-dev libmpfr-dev ocl-icd-opencl-dev libfmt-dev libmsgpack-dev python3-venv && $pip_install_2204"

prereq_array["ubuntu2404"]="apt-get -y update && apt-get install -y git cmake wget vim openssl libssl-dev libelf-dev kmod pciutils gcc g++ pkg-config libpci-dev libnuma-dev libffi-dev git libopenmpi-dev gawk mesa-common-dev libtool python3 texinfo libbison-dev bison flex libbabeltrace-dev python3-pip libncurses-dev liblzma-dev python3-setuptools python3-dev python3-barectf python3-pip python3-pip-whl python3-requests python3-venv python3-yaml libudev-dev libgmp-dev debianutils devscripts cli-common-dev rsync libsystemd-dev libdw-dev libgtest-dev libstdc++-12-dev sudo python3-lxml ccache libgmp-dev libmpfr-dev make ocl-icd-opencl-dev libfmt-dev libmsgpack-dev && $pip_install_2404"

prereq_array["centos7"]="yum install -y make gcc-c++ git cmake wget vim openssl-devel elfutils-libelf-devel pciutils-devel numactl-devel libffi-devel mesa-libGL-devel libtool texinfo bison flex ncurses-devel expat-devel xz-devel libbabeltrace-devel gmp-devel rpm-build rsync systemd-devel gtest-devel libpciaccess-devel elfutils-devel ccache libxml2-devel xz-lzma-compat devtoolset-9 devtoolset-9-libatomic-devel devtoolset-9-elfutils-libelf-devel scl-utils mpfr-devel gettext libcurl-devel ocl-icd-devel && yum remove -y python3*"

prereq_array["centos8"]="yum install -y dnf-plugins-core && yum config-manager --set-enabled PowerTools && yum install -y gcc-c++ git cmake wget vim openssl-devel elfutils-libelf-devel pciutils-devel numactl-devel libffi-devel mesa-libGL-devel libtool texinfo bison flex ncurses-devel expat-devel xz-devel libbabeltrace-devel gmp-devel rpm-build rsync systemd-devel gtest-devel elfutils-devel ccache python38 python38-devel mpfr-devel ocl-icd-devel && yum remove -y python36* && $pip_install"

prereq_array["centos9"]="yum install -y dnf-plugins-core gcc-c++ git cmake wget vim openssl-devel elfutils-libelf-devel pciutils-devel numactl-devel libffi-devel mesa-libGL-devel libtool texinfo bison flex ncurses-devel expat-devel xz-devel libbabeltrace-devel gmp-devel rpm-build rsync systemd-devel gtest-devel ccache mpfr-devel ocl-icd-devel && $pip_install"

prereq_array["rhel8"]="yum update -y && yum install -y dnf-plugins-core && yum install -y gcc-c++ git cmake wget vim openssl-devel elfutils-libelf-devel pciutils-devel numactl-devel libffi-devel mesa-libGL-devel libtool texinfo bison flex ncurses-devel expat-devel xz-devel libbabeltrace-devel gmp-devel rpm-build rsync systemd-devel gtest-devel elfutils-devel ccache python38 python38-devel mpfr-devel ocl-icd-devel libatomic libquadmath-devel msgpack-devel fmt-devel && $pip_install"

prereq_array["rhel9"]="dnf -y update && dnf -y install dnf-plugins-core && dnf -y install gdb gcc-c++ git cmake wget vim openssl-devel elfutils-libelf-devel pciutils-devel numactl-devel libffi-devel mesa-libGL-devel libtool texinfo bison flex ncurses-devel expat-devel xz-devel libbabeltrace-devel gmp-devel rpm-build rsync systemd-devel gtest-devel elfutils-devel ccache python3-devel mpfr-devel ocl-icd-devel libatomic libquadmath-devel msgpack-devel fmt-devel && $pip_install"

prereq_array["sles15"]="zypper install -y which cmake wget vim libopenssl-devel elfutils libelf-devel git pciutils-devel libffi-devel gcc gcc-c++ libnuma-devel openmpi2-devel Mesa-libGL-devel libquadmath0 libtool texinfo bison flex babeltrace-devel python3 python3-pip python3-devel python3-setuptools makeinfo libexpat-devel xz-devel gmp-devel rpm-build rsync libdrm-devel libX11-devel systemd-devel libdw-devel hwdata unzip ccache mpfr-devel ocl-icd-devel msgpack-devel fmt-devel gcc7-fortran; $pip_install"

# Some prep
default_os="ubuntu2404 ubuntu2204 rhel8 rhel9 sles15"
OS=${OS:-$default_os}
export DOCKER_HOME=/home/release; export DOCKER_AOMP=/usr/lib/aomp; export DOCKER_AOMP_REPOS=/home/release/git/aomp$AOMP_VERSION
exports="export HOME=/home/release; export AOMP=/usr/lib/aomp; export AOMP_REPOS=/home/release/git/aomp$AOMP_VERSION; export AOMP_EXTERNAL_MANIFEST=1; export AOMP_JOB_THREADS=128; export AOMP_SKIP_FLANG_NEW=0"

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
  getcontainer
  docker exec -i $docker_name /bin/bash -c "mkdir -p /home/release/git/aomp$AOMP_VERSION"

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
  elif [ "$system" == "rhel8" ]; then
    docker exec -i $docker_name /bin/bash -c "sed -i 's%^enabled = .*%enabled = 0%' /etc/yum.repos.d/ubi.repo"
    docker cp $DOCKERX_HOST/rhel8.repo $container:/etc/yum.repos.d/redhat-partner.repo
    docker exec -i $docker_name /bin/bash -c "yum -y install https://dl.fedoraproject.org/pub/epel/epel-release-latest-8.noarch.rpm && rpm -ql epel-release"
  elif [ "$system" == "rhel9" ]; then
    docker exec -i $docker_name /bin/bash -c "sed -i 's%^enabled = .*%enabled = 0%' /etc/yum.repos.d/ubi.repo"
    docker cp $DOCKERX_HOST/rhel9.repo $container:/etc/yum.repos.d/redhat-partner.repo
    docker exec -i $docker_name /bin/bash -c "dnf install -y  https://dl.fedoraproject.org/pub/epel/epel-release-latest-9.noarch.rpm https://dl.fedoraproject.org/pub/epel/epel-next-release-latest-9.noarch.rpm && rpm -ql epel-release"
    docker exec -i $docker_name /bin/bash -c "echo 'timeout=300' >> /etc/yum.conf"
  fi

  # Setup directory structure
  docker exec -i $docker_name /bin/bash -c "$exports; mkdir -p $DOCKER_AOMP_REPOS; mkdir -p $DOCKER_HOME/logs"

  # Hardcode timezone for tzdata install to avoid an interactive prompt
  docker exec -i $docker_name /bin/bash -c "$exports; ln -fs /usr/share/zoneinfo/America/Chicago /etc/localtime"

  # Install prerequisite system packages
  if [ "$system" == "sles15" ]; then
    set +e
    docker exec -i $docker_name /bin/bash -c "zypper refresh"
    docker exec -i $docker_name /bin/bash -c "zypper addrepo https://download.opensuse.org/repositories/science/SLE_15_SP5/science.repo"
    docker exec -i $docker_name /bin/bash -c "zypper addrepo https://download.opensuse.org/repositories/openSUSE:/Backports:/SLE-15-SP4/standard/openSUSE:Backports:SLE-15-SP4.repo"
    docker exec -i $docker_name /bin/bash -c "zypper addrepo https://download.opensuse.org/repositories/openSUSE:/Backports:/SLE-15-SP3/standard/openSUSE:Backports:SLE-15-SP3.repo && zypper --gpg-auto-import-keys refresh"
    docker exec -i $docker_name /bin/bash -c "$exports; ${prereq_array[$system]} 2>&1 | tee $DOCKER_HOME/logs/$system-preq.out"
    set -e
    docker exec -i $docker_name /bin/bash -c "zypper install -y --force libncurses6=6.1-150000.5.15.1; zypper install -y ncurses-devel"
    docker exec -i $docker_name /bin/bash -c "mkdir /tmp/googletest; cd /tmp/googletest; git clone https://github.com/google/googletest; cd googletest; git checkout release-1.14.0; mkdir build; cd build; cmake -DCMAKE_POSITION_INDEPENDENT_CODE=ON ..; make -j16; make -j16 install"
  else
    docker exec -i $docker_name /bin/bash -c "$exports; DEBIAN_FRONTEND=noninteractive ${prereq_array[$system]} 2>&1 | tee -a $DOCKER_HOME/logs/$system-preq.out"
  fi
  if [ "$system" == "centos7" ]; then
    exports="$exports; source /opt/rh/devtoolset-9/enable"
    docker exec -i $docker_name /bin/bash -c "$exports; cd /home/release; wget https://www.python.org/ftp/python/3.8.13/Python-3.8.13.tgz; tar xf Python-3.8.13.tgz; cd Python-3.8.13; ./configure --enable-optimizations --enable-shared; make altinstall; ln -s /usr/local/bin/python3.8 /usr/bin/python3; $pip_install_centos7"
    docker exec -i $docker_name /bin/bash -c "$exports; cd /home/release; wget https://github.com/git/git/archive/refs/tags/v2.19.0.tar.gz; tar xzf v2.19.0.tar.gz; cd git-2.19.0; make -j16 prefix=/usr/local install"
  fi

  # Run build_prerequisites.sh to build cmake, hwloc, rocmsmi, etc
  docker exec -i $docker_name /bin/bash -c "$exports; cd $DOCKER_AOMP_REPOS; git clone -b aomp-$AOMP_VERSION_STRING https://github.com/ROCm-Developer-Tools/aomp; cd aomp/bin; ./build_prereq.sh 2>&1 | tee $DOCKER_HOME/logs/$system-prereq.out"

  # Clone repos
  docker exec -i $docker_name /bin/bash -c "$exports; cd $DOCKER_AOMP_REPOS/aomp/bin; SINGLE_BRANCH=1 ./clone_aomp.sh 2>&1 | tee $DOCKER_HOME/logs/$system-clone.out"
  if [ "$AOMP_HIP_LIBRARIES" == "1" ]; then
    docker exec -i $docker_name /bin/bash -c "$exports; cd $DOCKER_AOMP_REPOS/aomp/bin/rocmlibs; SINGLE_BRANCH=1 ./clone_rocmlibs.sh 2>&1 | tee $DOCKER_HOME/logs/$system-clone-rocmlibs.out"
  fi
}

function build(){
  if [ "$system" == "ubuntu2404" ]; then
    exports="$exports; PATH=/opt/venv/bin:$PATH; python3 -m venv /opt/venv"
  fi
  docker exec -i $docker_name /bin/bash -c "$exports; cd $DOCKER_AOMP_REPOS/aomp/bin; ./build_aomp.sh 2>&1 | tee $DOCKER_HOME/logs/$system-build.out"
  if [ "$AOMP_HIP_LIBRARIES" == "1" ]; then
    docker exec -i $docker_name /bin/bash -c "$exports; cd $DOCKER_AOMP_REPOS/aomp/bin/rocmlibs; ./build_rocmlibs.sh 2>&1 | tee $DOCKER_HOME/logs/$system-build-rocmlibs.out"
  fi
}

function package(){
  getcontainer
  docker exec -i $docker_name /bin/bash -c "grep 'END build_aomp' $DOCKER_HOME/logs/$system-build.out"
  if [ "$?" -eq 0 ]; then
    if [[ "$system" =~ "ubuntu" ]]; then
      # Update changelog with user patch
      docker exec -i $docker_name /bin/bash -c "$exports; cd $DOCKER_AOMP_REPOS/aomp/bin; patch debian/changelog /dockerx/changelog.patch"
      # Build aomp debian
      docker exec -i $docker_name /bin/bash -c "$exports; cd $DOCKER_AOMP_REPOS/aomp/bin; ./build_fixups.sh; DOCKER=1 ./build-deb-aomp.sh 2>&1 | tee $DOCKER_HOME/logs/$system-package.out; git checkout debian/changelog"

      if [ "$AOMP_HIP_LIBRARIES" == "1" ]; then
        # Build aomp-hip-libraries debian
        docker exec -i $docker_name /bin/bash -c "$exports; cd $DOCKER_AOMP_REPOS/aomp/bin; ./build_fixups.sh; DOCKER=1 ./build-deb-aomp.sh aomp-hip-libraries 2>&1 | tee $DOCKER_HOME/logs/$system-package-hip-libraries.out"
      fi
      # Copy to host
      docker cp $container:/tmp/build-deb/debs/. $host_packages
    else
      # Update changelog with user patch and change aomp version header
      docker exec -i $docker_name /bin/bash -c "$exports; cd $DOCKER_AOMP_REPOS/aomp/bin; patch debian/changelog /dockerx/changelog.patch; sed -i -e 's/aomp (.*)/aomp ($AOMP_VERSION_STRING)/g' debian/changelog"
      # Build aomp rpm
      if [ "$system" == "centos7" ]; then
        docker exec -i $docker_name /bin/bash -c "$exports; cd $DOCKER_AOMP_REPOS/aomp/bin; ./build_fixups.sh; DOCKER=1 ./build-rpm.sh aomp_CENTOS_7 2>&1 | tee $DOCKER_HOME/logs/$system-package.out"
      else
        docker exec -i $docker_name /bin/bash -c "$exports; cd $DOCKER_AOMP_REPOS/aomp/bin; ./build_fixups.sh; DOCKER=1 ./build-rpm.sh 2>&1 | tee $DOCKER_HOME/logs/$system-package.out"
        # Copy to host
        docker cp $container:/tmp/home/rpmbuild/RPMS/x86_64/. $host_packages
        # Build aomp-hip-libraries rpm
        if [ "$AOMP_HIP_LIBRARIES" == "1" ]; then
          docker exec -i $docker_name /bin/bash -c "$exports; cd $DOCKER_AOMP_REPOS/aomp/bin; ./build_fixups.sh; DOCKER=1 ./build-rpm.sh aomp-hip-libraries 2>&1 | tee $DOCKER_HOME/logs/$system-package-hip-libraries.out"
          # Copy to host
          docker cp $container:/tmp/home/rpmbuild/RPMS/x86_64/. $host_packages
        fi
      fi
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
      echo "OS options: ubuntu2204, ubuntu2404, rhel8, rhel9, sles15"
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
