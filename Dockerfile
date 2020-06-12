ARG version=20.04
FROM ubuntu:$version
ARG DEBIAN_FRONTEND=noninteractive

# hadolint ignore=DL3008
RUN apt-get update \
  && apt-get install -y --no-install-recommends software-properties-common \
  && add-apt-repository -y ppa:git-core/ppa \
  && apt-get update \
  && apt-get install -y --no-install-recommends \
    bzip2 \
    ca-certificates \
    curl \
    file \
    fonts-dejavu-core \
    g++ \
    git \
    less \
    libz-dev \
    locales \
    make \
    netbase \
    openssh-client \
    patch \
    sudo \
    uuid-runtime \
    tzdata \
  && rm -rf /var/lib/apt/lists/* \
  && localedef -i en_US -f UTF-8 en_US.UTF-8 \
  && useradd -m -s /bin/bash linuxbrew \
  && echo 'linuxbrew ALL=(ALL) NOPASSWD:ALL' >>/etc/sudoers

COPY . /home/linuxbrew/.linuxbrew/Homebrew
ENV PATH=/home/linuxbrew/.linuxbrew/bin:/home/linuxbrew/.linuxbrew/sbin:$PATH
WORKDIR /home/linuxbrew

# hadolint ignore=DL3003
RUN cd /home/linuxbrew/.linuxbrew \
  && mkdir -p bin etc include lib opt sbin share var/homebrew/linked Cellar \
  && ln -s ../Homebrew/bin/brew /home/linuxbrew/.linuxbrew/bin/ \
  && git -C /home/linuxbrew/.linuxbrew/Homebrew remote set-url origin https://github.com/Homebrew/brew \
  && git -C /home/linuxbrew/.linuxbrew/Homebrew fetch origin \
  && HOMEBREW_NO_ANALYTICS=1 HOMEBREW_NO_AUTO_UPDATE=1 brew tap homebrew/core \
  && brew install-bundler-gems \
  && brew cleanup \
  && { git -C /home/linuxbrew/.linuxbrew/Homebrew config --unset gc.auto; true; } \
  && { git -C /home/linuxbrew/.linuxbrew/Homebrew config --unset homebrew.devcmdrun; true; } \
  && rm -rf ~/.cache \
  && chown -R linuxbrew: /home/linuxbrew/.linuxbrew \
  && chmod -R g+w,o-w /home/linuxbrew/.linuxbrew
