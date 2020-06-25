# 0.0.1-API-Parser

Parser for fixing this: https://github.com/Homebrew/brew/issues/5725

## Overview

Homebrew is used to install software (packages). Homebrew uses 'formulae' to determine how a package is installed.
This project will automatically check which packages have had newer versions released, whether the package has an open PR on homebrew, and display the results.

## High-level Solution

- Fetch latest package version information from [repology.org](https://repology.org/) and store on file system.
- Fetch Homebrew Formulae information from [HomeBrew Formulae](https://formulae.brew.sh)
- Compare Current Homebrew Formulae version numbers and those coming from Repology's API and Livecheck.
- Determine whether package has open PR.
- Display results.

## Details

- This project can be run automatically at set intervals via GitHub Actions.
- Executing `ruby printPackageUpdates.rb` from the command line will query
  both the Repology and Homebrew APIs. Homebrew's current version of each
  package will be compared to the latest version of the package, per Repology's response.
- Homebrew's livecheck is also queried for each package, and that data is parsed, if available.
- Checks whether there is open PR for package.
- Each outdated package will be displayed to the console like so:
- Note that some packages will not be included in the Livecheck response.  Those will have a 'Livecheck latest:' value of 'Not found'.

```
Package: openclonk
Brew current: 7.0
Repology latest: 8.1
Livecheck latest: 8.1
Has Open PR?: true

Package: openjdk
Brew current: 13.0.2+8
Repology latest: 15.0.0.0~14
Livecheck latest: Not found.
Has Open PR?: false

Package: opentsdb
Brew current: 2.3.1
Repology latest: 2.4.0
Livecheck latest: 2.4.0
Has Open PR?: true
```
