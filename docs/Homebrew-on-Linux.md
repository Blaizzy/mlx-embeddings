---
title: Homebrew on Linux
logo: https://brew.sh/assets/img/linuxbrew.png
image: https://brew.sh/assets/img/linuxbrew.png
redirect_from:
  - /linux
  - /Linux
  - /Linuxbrew
---
The Homebrew package manager may be used on Linux and [Windows Subsystem for Linux (WSL)](https://docs.microsoft.com/en-us/windows/wsl/about). Homebrew was formerly referred to as Linuxbrew when running on Linux or WSL. It can be installed in your home directory, in which case it does not use *sudo*. Homebrew does not use any libraries provided by your host system, except *glibc* and *gcc* if they are new enough. Homebrew can install its own current versions of *glibc* and *gcc* for older distributions of Linux.

[Features](#features), [dependencies](#dependencies) and [installation instructions](#install) are described below. Terminology (e.g. the difference between a Cellar, Tap, Cask and so forth) is [explained in the documentation](Formula-Cookbook.md#homebrew-terminology).

## Features

+ Can install software to your home directory and so does not require *sudo*
+ Install software not packaged by your host distribution
+ Install up-to-date versions of software when your host distribution is old
+ Use the same package manager to manage your macOS, Linux, and Windows systems

## Install

Paste at a terminal prompt:

```sh
sh -c "$(curl -fsSL https://raw.githubusercontent.com/Linuxbrew/install/master/install.sh)"
```

The installation script installs Homebrew to `/home/linuxbrew/.linuxbrew` using *sudo* if possible and in your home directory at `~/.linuxbrew` otherwise. Homebrew does not use *sudo* after installation. Using `/home/linuxbrew/.linuxbrew` allows the use of more binary packages (bottles) than installing in your personal home directory.

Follow the *Next steps* instructions to add Homebrew to your `PATH` and to your bash shell profile script, either `~/.profile` on Debian/Ubuntu or `~/.bash_profile` on CentOS/Fedora/RedHat.

```sh
test -d ~/.linuxbrew && eval $(~/.linuxbrew/bin/brew shellenv)
test -d /home/linuxbrew/.linuxbrew && eval $(/home/linuxbrew/.linuxbrew/bin/brew shellenv)
test -r ~/.bash_profile && echo "eval \$($(brew --prefix)/bin/brew shellenv)" >>~/.bash_profile
echo "eval \$($(brew --prefix)/bin/brew shellenv)" >>~/.profile
```

You're done! Try installing a package:

```sh
brew install hello
```

If you're using an older distribution of Linux, installing your first package will also install a recent version of `glibc` and `gcc`. Use `brew doctor` to troubleshoot common issues.

## Linux/WSL Requirements

+ **GCC** 4.7.0 or newer
+ **Linux** 2.6.32 or newer
+ **Glibc** 2.13 or newer
+ **64-bit x86_64** CPU

Paste at a terminal prompt:

### Debian or Ubuntu

```sh
sudo apt-get install build-essential curl file git
```

### Fedora, CentOS, or Red Hat

```sh
sudo yum groupinstall 'Development Tools'
sudo yum install curl file git
sudo yum install libxcrypt-compat # needed by Fedora 30 and up
```

### ARM

Homebrew can run on 32-bit ARM (Raspberry Pi and others) and 64-bit ARM (AArch64), but no binary packages (bottles) are available. Support for ARM is on a best-effort basis. Pull requests are welcome to improve the experience on ARM platforms.

### 32-bit x86

Homebrew does not currently support 32-bit x86 platforms. It would be possible for Homebrew to work on 32-bit x86 platforms with some effort. An interested and dedicated person could maintain a fork of Homebrew to develop support for 32-bit x86.

## Alternative Installation

Extract or `git clone` Homebrew wherever you want. Use `/home/linuxbrew/.linuxbrew` if possible (to enable the use of binary packages).

```sh
git clone https://github.com/Homebrew/brew ~/.linuxbrew/Homebrew
mkdir ~/.linuxbrew/bin
ln -s ~/.linuxbrew/Homebrew/bin/brew ~/.linuxbrew/bin
eval $(~/.linuxbrew/bin/brew shellenv)
```

## Homebrew on Linux Community

- [@HomebrewOnLinux on Twitter](https://twitter.com/HomebrewOnLinux)
- [Homebrew/linuxbrew-core on GitHub](https://github.com/Homebrew/linuxbrew-core)
- [Homebrew on Linux category](https://discourse.brew.sh/c/linuxbrew) of [Homebrew's Discourse](https://discourse.brew.sh)

## Sponsors

Our binary packages (bottles) are built on [CircleCI](https://circleci.com/) and hosted by [Bintray](https://bintray.com/linuxbrew).

[<img alt="CircleCI logo" style="height:1in" src="https://assets.brandfolder.com/otz6k5-cj8pew-e4rk9u/original/circle-logo-horizontal-black.png">](https://circleci.com/)

[![Downloads by Bintray](https://bintray.com/docs/images/downloads_by_bintray_96.png)](https://bintray.com/linuxbrew)
