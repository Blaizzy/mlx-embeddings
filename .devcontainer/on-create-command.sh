#!/bin/bash
set -e

# dump information variables for debugging
echo "==> env"
env | grep -v TOKEN
echo
echo "==> /etc/os-release"
cat /etc/os-release || true
echo
echo "==> /etc/lsb-release"
cat /etc/lsb-release || true
echo
echo "==> /etc/issue"
cat /etc/issue || true
echo

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
APT_GET_INSTALL="openssh-server zsh"

# Ubuntu 18.04 doesn't include zsh-autosuggestions
if ! grep -q "Ubuntu 18.04" /etc/issue &>/dev/null
then
  APT_GET_INSTALL="$APT_GET_INSTALL zsh-autosuggestions"
fi

sudo apt-get install -y \
  -o Dpkg::Options::=--force-confdef \
  -o Dpkg::Options::=--force-confnew \
  $APT_GET_INSTALL

# Start the SSH server so that `gh cs ssh` works.
sudo service ssh start
