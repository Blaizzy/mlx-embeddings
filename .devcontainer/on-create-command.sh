#!/bin/bash
set -e

# fix permissions so Homebrew and Bundler don't complain
sudo chmod -R g-w,o-w /home/linuxbrew

# install Homebrew's development gems
brew install-bundler-gems --groups=sorbet

# install Homebrew formulae we might need
brew install shellcheck shfmt gh gnu-tar

# cleanup any mess
brew cleanup

# install some useful development things
sudo apt-get update
sudo apt-get install -y \
  -o Dpkg::Options::=--force-confdef \
  -o Dpkg::Options::=--force-confnew \
  openssh-server \
  zsh \
  zsh-autosuggestions

# Start the SSH server so that `gh cs ssh` works.
sudo service ssh start
