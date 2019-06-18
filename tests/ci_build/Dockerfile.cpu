FROM centos:6
LABEL maintainer "DMLC"

# Environment
ENV DEBIAN_FRONTEND noninteractive

# Install all basic requirements
RUN \
    yum -y update && \
    yum install -y tar unzip wget xz git && \
    wget http://people.centos.org/tru/devtools-2/devtools-2.repo -O /etc/yum.repos.d/devtools-2.repo && \
    yum install -y devtoolset-2-gcc devtoolset-2-binutils devtoolset-2-gcc-c++ && \
    # Python
    wget https://repo.continuum.io/miniconda/Miniconda3-4.5.12-Linux-x86_64.sh && \
    bash Miniconda3-4.5.12-Linux-x86_64.sh -b -p /opt/python && \
    # CMake
    wget -nv -nc https://cmake.org/files/v3.2/cmake-3.2.0-Linux-x86_64.sh --no-check-certificate && \
    bash cmake-3.2.0-Linux-x86_64.sh --skip-license --prefix=/usr

ENV PATH=/opt/python/bin:$PATH
ENV CC=/opt/rh/devtoolset-2/root/usr/bin/gcc
ENV CXX=/opt/rh/devtoolset-2/root/usr/bin/c++
ENV CPP=/opt/rh/devtoolset-2/root/usr/bin/cpp

# Install Python packages
RUN \
    pip install numpy pytest scipy scikit-learn wheel pandas

ENV GOSU_VERSION 1.10

# Install lightweight sudo (not bound to TTY)
RUN set -ex; \
    wget -O /usr/local/bin/gosu "https://github.com/tianon/gosu/releases/download/$GOSU_VERSION/gosu-amd64" && \
    chmod +x /usr/local/bin/gosu && \
    gosu nobody true

# Default entry-point to use if running locally
# It will preserve attributes of created files
COPY entrypoint.sh /scripts/

WORKDIR /workspace
ENTRYPOINT ["/scripts/entrypoint.sh"]
