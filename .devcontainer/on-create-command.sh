#!/bin/bash
set -e

sudo apt-get update
sudo apt-get install -y \
  -o Dpkg::Options::=--force-confdef \
  -o Dpkg::Options::=--force-confnew \
  zsh \
  shellcheck \
  zsh-autosuggestions
