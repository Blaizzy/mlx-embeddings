#!/bin/bash

set -e

# silence bundler complaining about being root
mkdir ~/.bundle
echo 'BUNDLE_SILENCE_ROOT_WARNING: "1"' > ~/.bundle/config

# configure git
git config --global user.name "BrewTestBot"
git config --global user.email "homebrew-test-bot@lists.sfconservancy.org"

# setup SSH
mkdir ~/.ssh
chmod 700 ~/.ssh
echo "$RUBYDOC_DEPLOY_KEY" > ~/.ssh/id_ed25519
chmod 600 ~/.ssh/id_ed25519
git config --global core.sshCommand "ssh -i ~/.ssh/id_ed25519 -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no"

# clone rubydoc.brew.sh with SSH so we can push back
git clone git@github.com:Homebrew/rubydoc.brew.sh
cd rubydoc.brew.sh

# clone latest Homebrew/brew
git clone --depth=1 https://github.com/Homebrew/brew

# run rake to build documentation
gem install bundler
bundle install
bundle exec rake

# commit and push generated files
git add docs
git diff --exit-code HEAD -- docs && exit 0
git commit -m 'docs: update from Homebrew/brew push' docs
git push
