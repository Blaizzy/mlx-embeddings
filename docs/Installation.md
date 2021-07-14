# Installation

Instructions for a supported install of Homebrew are on the [homepage](https://brew.sh).

This script installs Homebrew to its preferred prefix (`/usr/local`
for macOS Intel, `/opt/homebrew` for Apple Silicon and `/home/linuxbrew/.linuxbrew` for Linux) so that
[you don’t need sudo](FAQ.md#why-does-homebrew-say-sudo-is-bad) when you
`brew install`. It is a careful script; it can be run even if you have stuff
installed in the preferred prefix already. It tells you exactly what it will do before
it does it too. You have to confirm everything it will do before it starts.

## macOS Requirements

* A 64-bit Intel CPU or Apple Silicon CPU <sup>[1](#1)</sup>
* macOS Mojave (10.14) (or higher) <sup>[2](#2)</sup>
* Command Line Tools (CLT) for Xcode: `xcode-select --install`,
  [developer.apple.com/downloads](https://developer.apple.com/downloads) or
  [Xcode](https://itunes.apple.com/us/app/xcode/id497799835) <sup>[3](#3)</sup>
* A Bourne-compatible shell for installation (e.g. `bash` or `zsh`) <sup>[4](#4)</sup>

## Git Remote Mirroring

You can set `HOMEBREW_BREW_GIT_REMOTE` and/or `HOMEBREW_CORE_GIT_REMOTE` in your shell environment to use geolocalized Git mirrors to speed up Homebrew's installation with this script and, after installation, `brew update`.

```bash
export HOMEBREW_BREW_GIT_REMOTE="..."  # put your Git mirror of Homebrew/brew here
export HOMEBREW_CORE_GIT_REMOTE="..."  # put your Git mirror of Homebrew/homebrew-core here
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install.sh)"
```

The default Git remote will be used if the corresponding environment variable is unset.

## Alternative Installs

### Linux or Windows 10 Subsystem for Linux

Check out [the Homebrew on Linux installation documentation](Homebrew-on-Linux.md).

### Untar anywhere

Just extract (or `git clone`) Homebrew wherever you want. Just avoid:

* Directories with names that contain spaces. Homebrew itself can handle spaces, but many build scripts cannot.
* `/tmp` subdirectories because Homebrew gets upset.
* `/sw` and `/opt/local` because build scripts get confused when Homebrew is there instead of Fink or MacPorts, respectively.

However do yourself a favour and use the installer to install to the default prefix. Some things may
not build when installed elsewhere. One of the reasons Homebrew just
works relative to the competition is **because** we recommend installing
here. *Pick another prefix at your peril!*

```sh
mkdir homebrew && curl -L https://github.com/Homebrew/brew/tarball/master | tar xz --strip 1 -C homebrew
```

### Multiple installations

Create a Homebrew installation wherever you extract the tarball. Whichever `brew` command is called is where the packages will be installed. You can use this as you see fit, e.g. a system set of libs in the default prefix and tweaked formulae for development in `~/homebrew`.

## Uninstallation

Uninstallation is documented in the [FAQ](FAQ.md).

<a name="1"><sup>1</sup></a> For 32-bit or PPC support see
[Tigerbrew](https://github.com/mistydemeo/tigerbrew).

<a name="2"><sup>2</sup></a> 10.14 or higher is recommended. 10.9–10.13 are
supported on a best-effort basis. For 10.4-10.6 see
[Tigerbrew](https://github.com/mistydemeo/tigerbrew).

<a name="3"><sup>3</sup></a> Most formulae require a compiler. A handful
require a full Xcode installation. You can install Xcode, the CLT, or both;
Homebrew supports all three configurations. Downloading Xcode may require an
Apple Developer account on older versions of Mac OS X. Sign up for free
[here](https://developer.apple.com/register/index.action).

<a name="4"><sup>4</sup></a> The one-liner installation method found on
[brew.sh](https://brew.sh) requires a Bourne-compatible shell (e.g. bash or
zsh). Notably, fish, tcsh and csh will not work.
