#!/bin/bash
set -e

# fix permissions so Homebrew and Bundler don't complain
sudo chmod -R g-w,o-w /home/linuxbrew

# everything below is too slow to do unless prebuilding so skip it
CODESPACES_ACTION_NAME="$(cat /workspaces/.codespaces/shared/environment-variables.json | jq -r '.ACTION_NAME')"
if [ "$CODESPACES_ACTION_NAME" != "createPrebuildTemplate" ];
  echo "Skipping slow items, not prebuilding."
  exit 0
fi

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
