FROM registry.access.redhat.com/ubi9/ubi:9.4

# ROCm release we want to run
COPY rocm-6.3.repo /etc/yum.repos.d/rocm.repo
RUN yum clean packages && yum clean all

# Other utilities we need
COPY prerequisites /tmp/prerequisites
RUN yum install --disableplugin=subscription-manager -y https://dl.fedoraproject.org/pub/epel/epel-release-latest-9.noarch.rpm
RUN yum update --disableplugin=subscription-manager -y
RUN yum install --disableplugin=subscription-manager -y $(cat /tmp/prerequisites)

# Install ROCm
RUN yum install --disableplugin=subscription-manager -y rocm-device-libs rocm-core rocminfo hsa-rocr-dev

# Update the group id inside the container to match host (and get access to GPU)
RUN groupmod -g 109 render

# Create buildbot worker user inside the container
RUN useradd --create-home -G video,render --shell /bin/bash botworker
