brew(1) -- The Missing Package Manager for macOS
================================================

## SYNOPSIS

`brew` `--version`<br>
`brew` *`command`* [`--verbose`|`-v`] [*`options`*] [*`formula`*] ...

## DESCRIPTION

Homebrew is the easiest and most flexible way to install the UNIX tools Apple
didn't include with macOS.

## ESSENTIAL COMMANDS

For the full command list, see the [COMMANDS](#commands) section.

With `--verbose` or `--debug`, many commands print extra debugging information. Note that
these options should only appear after a command.

### `install` *`formula`*:

Install *`formula`*.

*`formula`* is usually the name of the formula to install, but it has other syntaxes which
are listed in the [SPECIFYING FORMULAE](#specifying-formulae) section.

### `uninstall` *`formula`*:

Uninstall *`formula`*.

### `list`:

List all installed formulae.

### `search` (*`text`*|`/`*`text`*`/`):

Perform a substring search of cask tokens and formula names for *`text`*. If *`text`*
is flanked by slashes, it is interpreted as a regular expression.
The search for *`text`* is extended online to `homebrew/core` and `homebrew/cask`.
If no search term is provided, all locally available formulae are listed.

## COMMANDS

### `analytics` [*`subcommand`*]

If `on` or `off` is passed, turn Homebrew's analytics on or off respectively.

If `state` is passed, display the current anonymous user behaviour analytics
state. Read more at <https://docs.brew.sh/Analytics>.

If `regenerate-uuid` is passed, regenerate the UUID used in Homebrew's
analytics.

### `cat` *`formula`*

Display the source of *`formula`*.

### `cleanup` [*`options`*] [*`formula`*|*`cask`*]

Remove stale lock files and outdated downloads for all formulae and casks, and
remove old versions of installed formulae. If arguments are specified, only do
this for the given formulae and casks.

* `--prune`:
  Remove all cache files older than specified *`days`*.
* `-n`, `--dry-run`:
  Show what would be removed, but do not actually remove anything.
* `-s`:
  Scrub the cache, including downloads for even the latest versions. Note downloads for any installed formulae or casks will still not be deleted. If you want to delete those too: `rm -rf "$(brew --cache)"`
* `--prune-prefix`:
  Only prune the symlinks and directories from the prefix and remove no other files.

### `command` *`cmd`*

Display the path to the file being used when invoking `brew` *`cmd`*.

### `commands` [*`options`*]

Show lists of built-in and external commands.

* `-q`, `--quiet`:
  List only the names of commands without category headers.
* `--include-aliases`:
  Include aliases of internal commands.

### `config`

Show Homebrew and system configuration info useful for debugging. If you file a
bug report, you will be required to provide this information.

### `deps` [*`options`*] [*`formula`*]

Show dependencies for *`formula`*. Additional options specific to *`formula`* may be
appended to the command. When given multiple formula arguments, show the
intersection of dependencies for each formula.

* `-n`:
  Sort dependencies in topological order.
* `--1`:
  Only show dependencies one level down, instead of recursing.
* `--union`:
  Show the union of dependencies for multiple *`formula`*, instead of the intersection.
* `--full-name`:
  List dependencies by their full name.
* `--include-build`:
  Include `:build` dependencies for *`formula`*.
* `--include-optional`:
  Include `:optional` dependencies for *`formula`*.
* `--include-test`:
  Include `:test` dependencies for *`formula`* (non-recursive).
* `--skip-recommended`:
  Skip `:recommended` dependencies for *`formula`*.
* `--include-requirements`:
  Include requirements in addition to dependencies for *`formula`*.
* `--tree`:
  Show dependencies as a tree. When given multiple formula arguments, show individual trees for each formula.
* `--annotate`:
  Mark any build, test, optional, or recommended dependencies as such in the output.
* `--installed`:
  List dependencies for formulae that are currently installed. If *`formula`* is specified, list only its dependencies that are currently installed.
* `--all`:
  List dependencies for all available formulae.
* `--for-each`:
  Switch into the mode used by the `--all` option, but only list dependencies for each provided *`formula`*, one formula per line. This is used for debugging the `--installed`/`--all` display mode.

### `desc` [*`options`*] (*`text`*|`/`*`text`*`/`|*`formula`*)

Display *`formula`*'s name and one-line description. Formula descriptions are
cached; the cache is created on the first search, making that search slower than
subsequent ones.

* `-s`, `--search`:
  Search both names and descriptions for *`text`*. If *`text`* is flanked by slashes, it is interpreted as a regular expression.
* `-n`, `--name`:
  Search just names for *`text`*. If *`text`* is flanked by slashes, it is interpreted as a regular expression.
* `-d`, `--description`:
  Search just descriptions for *`text`*. If *`text`* is flanked by slashes, it is interpreted as a regular expression.

### `diy` [*`options`*]

Automatically determine the installation prefix for non-Homebrew software. Using
the output from this command, you can install your own software into the Cellar
and then link it into Homebrew's prefix with `brew link`.

* `--name`:
  Explicitly set the *`name`* of the package being installed.
* `--version`:
  Explicitly set the *`version`* of the package being installed.

### `doctor` [*`options`*]

Check your system for potential problems. Will exit with a non-zero status if
any potential problems are found. Please note that these warnings are just used
to help the Homebrew maintainers with debugging if you file an issue. If
everything you use Homebrew for is working fine: please don't worry or file an
issue; just ignore this.

* `--list-checks`:
  List all audit methods, which can be run individually if provided as arguments.
* `-D`, `--audit-debug`:
  Enable debugging and profiling of audit methods.

### `fetch` [*`options`*] *`formula`*

Download a bottle (if available) or source packages for *`formula`*. For tarballs,
also print SHA-256 checksums.

* `--HEAD`:
  Fetch HEAD version instead of stable version.
* `--devel`:
  Fetch development version instead of stable version.
* `-f`, `--force`:
  Remove a previously cached version and re-fetch.
* `-v`, `--verbose`:
  Do a verbose VCS checkout, if the URL represents a VCS. This is useful for seeing if an existing VCS cache has been updated.
* `--retry`:
  Retry if downloading fails or re-download if the checksum of a previously cached version no longer matches.
* `--deps`:
  Also download dependencies for any listed *`formula`*.
* `-s`, `--build-from-source`:
  Download source packages rather than a bottle.
* `--build-bottle`:
  Download source packages (for eventual bottling) rather than a bottle.
* `--force-bottle`:
  Download a bottle if it exists for the current or newest version of macOS, even if it would not be used during installation.

### `gist-logs` [*`options`*] *`formula`*

Upload logs for a failed build of *`formula`* to a new Gist. Presents an error
message if no logs are found.

* `--with-hostname`:
  Include the hostname in the Gist.
* `-n`, `--new-issue`:
  Automatically create a new issue in the appropriate GitHub repository after creating the Gist.
* `-p`, `--private`:
  The Gist will be marked private and will not appear in listings but will be accessible with its link.

### `home` [*`formula`*]

Open *`formula`*'s homepage in a browser, or open Homebrew's own homepage if no
formula is provided.

### `info` [*`options`*] [*`formula`*]

Display brief statistics for your Homebrew installation.

If *`formula`* is provided, show summary of information about *`formula`*.

* `--analytics`:
  List global Homebrew analytics data or, if specified, installation and build error data for *`formula`* (provided neither `HOMEBREW_NO_ANALYTICS` nor `HOMEBREW_NO_GITHUB_API` are set).
* `--days`:
  How many days of analytics data to retrieve. The value for *`days`* must be `30`, `90` or `365`. The default is `30`.
* `--category`:
  Which type of analytics data to retrieve. The value for *`category`* must be `install`, `install-on-request` or `build-error`; `cask-install` or `os-version` may be specified if *`formula`* is not. The default is `install`.
* `--github`:
  Open the GitHub source page for *`formula`* in a browser. To view formula history locally: `brew log -p` *`formula`*
* `--json`:
  Print a JSON representation of *`formula`*. Currently the default and only accepted value for *`version`* is `v1`. See the docs for examples of using the JSON output: <https://docs.brew.sh/Querying-Brew>
* `--installed`:
  Print JSON of formulae that are currently installed.
* `--all`:
  Print JSON of all available formulae.
* `-v`, `--verbose`:
  Show more verbose analytics data for *`formula`*.

### `install` [*`options`*] *`formula`*

Install *`formula`*. Additional options specific to *`formula`* may be appended to
the command.

Unless `HOMEBREW_NO_INSTALL_CLEANUP` is set, `brew cleanup` will then be run for
the installed formulae or, every 30 days, for all formulae.

* `-d`, `--debug`:
  If brewing fails, open an interactive debugging session with access to IRB or a shell inside the temporary build directory.
* `--env`:
  If `std` is passed, use the standard build environment instead of superenv. If `super` is passed, use superenv even if the formula specifies the standard build environment.
* `--ignore-dependencies`:
  An unsupported Homebrew development flag to skip installing any dependencies of any kind. If the dependencies are not already present, the formula will have issues. If you're not developing Homebrew, consider adjusting your PATH rather than using this flag.
* `--only-dependencies`:
  Install the dependencies with specified options but do not install the formula itself.
* `--cc`:
  Attempt to compile using the specified *`compiler`*, which should be the name of the compiler's executable, e.g. `gcc-7` for GCC 7. In order to use LLVM's clang, specify `llvm_clang`. To use the Apple-provided clang, specify `clang`. This option will only accept compilers that are provided by Homebrew or bundled with macOS. Please do not file issues if you encounter errors while using this option.
* `-s`, `--build-from-source`:
  Compile *`formula`* from source even if a bottle is provided. Dependencies will still be installed from bottles if they are available.
* `--force-bottle`:
  Install from a bottle if it exists for the current or newest version of macOS, even if it would not normally be used for installation.
* `--include-test`:
  Install testing dependencies required to run `brew test` *`formula`*.
* `--devel`:
  If *`formula`* defines it, install the development version.
* `--HEAD`:
  If *`formula`* defines it, install the HEAD version, aka. master, trunk, unstable.
* `--fetch-HEAD`:
  Fetch the upstream repository to detect if the HEAD installation of the formula is outdated. Otherwise, the repository's HEAD will only be checked for updates when a new stable or development version has been released.
* `--keep-tmp`:
  Retain the temporary files created during installation.
* `--build-bottle`:
  Prepare the formula for eventual bottling during installation, skipping any post-install steps.
* `--bottle-arch`:
  Optimise bottles for the specified architecture rather than the oldest architecture supported by the version of macOS the bottles are built on.
* `-f`, `--force`:
  Install without checking for previously installed keg-only or non-migrated versions.
* `-v`, `--verbose`:
  Print the verification and postinstall steps.
* `--display-times`:
  Print install times for each formula at the end of the run.
* `-i`, `--interactive`:
  Download and patch *`formula`*, then open a shell. This allows the user to run `./configure --help` and otherwise determine how to turn the software package into a Homebrew package.
* `-g`, `--git`:
  Create a Git repository, useful for creating patches to the software.

### `leaves`

List installed formulae that are not dependencies of another installed formula.

### `link`, `ln` [*`options`*] *`formula`*

Symlink all of *`formula`*'s installed files into Homebrew's prefix. This is done
automatically when you install formulae but can be useful for DIY installations.

* `--overwrite`:
  Delete files that already exist in the prefix while linking.
* `-n`, `--dry-run`:
  List files which would be linked or deleted by `brew link --overwrite` without actually linking or deleting any files.
* `-f`, `--force`:
  Allow keg-only formulae to be linked.

### `list`, `ls` [*`options`*] [*`formula`*]

List all installed formulae.

If *`formula`* is provided, summarise the paths within its current keg.

* `--full-name`:
  Print formulae with fully-qualified names. If `--full-name` is not passed, other options (i.e. `-1`, `-l`, `-r` and `-t`) are passed to `ls`(1) which produces the actual output.
* `--unbrewed`:
  List files in Homebrew's prefix not installed by Homebrew.
* `--versions`:
  Show the version number for installed formulae, or only the specified formulae if *`formula`* are provided.
* `--multiple`:
  Only show formulae with multiple versions installed.
* `--pinned`:
  Show the versions of pinned formulae, or only the specified (pinned) formulae if *`formula`* are provided. See also `pin`, `unpin`.
* `-1`:
  Force output to be one entry per line. This is the default when output is not to a terminal.
* `-l`:
  List in long format. If the output is to a terminal, a total sum for all the file sizes is printed before the long listing.
* `-r`:
  Reverse the order of the sort to list the oldest entries first.
* `-t`:
  Sort by time modified, listing most recently modified first.

### `log` [*`options`*] [*`formula`*]

Show the `git log` for *`formula`*, or show the log for the Homebrew repository if
no formula is provided.

* `-p`, `--patch`:
  Also print patch from commit.
* `--stat`:
  Also print diffstat from commit.
* `--oneline`:
  Print only one line per commit.
* `-1`, `--max-count`:
  Print only one or a specified number of commits.

### `migrate` [*`options`*] *`formula`*

Migrate renamed packages to new names, where *`formula`* are old names of
packages.

* `-f`, `--force`:
  Treat installed *`formula`* and provided *`formula`* as if they are from the same taps and migrate them anyway.

### `missing` [*`options`*] [*`formula`*]

Check the given *`formula`* kegs for missing dependencies. If no *`formula`* are
provided, check all kegs. Will exit with a non-zero status if any kegs are found
to be missing dependencies.

* `--hide`:
  Act as if none of the specified *`hidden`* are installed. *`hidden`* should be a comma-separated list of formulae.

### `options` [*`options`*] [*`formula`*]

Show install options specific to *`formula`*.

* `--compact`:
  Show all options on a single line separated by spaces.
* `--installed`:
  Show options for formulae that are currently installed.
* `--all`:
  Show options for all available formulae.

### `outdated` [*`options`*] [*`formula`*]

List installed formulae that have an updated version available. By default,
version information is displayed in interactive shells, and suppressed
otherwise.

* `-q`, `--quiet`:
  List only the names of outdated kegs (takes precedence over `--verbose`).
* `-v`, `--verbose`:
  Include detailed version information.
* `--json`:
  Print output in JSON format. Currently the default and only accepted value for *`version`* is `v1`. See the docs for examples of using the JSON output: <https://docs.brew.sh/Querying-Brew>
* `--fetch-HEAD`:
  Fetch the upstream repository to detect if the HEAD installation of the formula is outdated. Otherwise, the repository's HEAD will only be checked for updates when a new stable or development version has been released.

### `pin` *`formula`*

Pin the specified *`formula`*, preventing them from being upgraded when issuing
the `brew upgrade` *`formula`* command. See also `unpin`.

### `postinstall` *`formula`*

Rerun the post-install steps for *`formula`*.

### `readall` [*`options`*] [*`tap`*]

Import all formulae from the specified *`tap`*, or from all installed taps if none
is provided. This can be useful for debugging issues across all formulae when
making significant changes to `formula.rb`, testing the performance of loading
all formulae or checking if any current formulae have Ruby issues.

* `--aliases`:
  Verify any alias symlinks in each tap.
* `--syntax`:
  Syntax-check all of Homebrew's Ruby files.

### `reinstall` [*`options`*] *`formula`*

Uninstall and then install *`formula`* using the same options it was originally
installed with, plus any appended brew formula options.

Unless `HOMEBREW_NO_INSTALL_CLEANUP` is set, `brew cleanup` will then be run for
the reinstalled formulae or, every 30 days, for all formulae.

* `-d`, `--debug`:
  If brewing fails, open an interactive debugging session with access to IRB or a shell inside the temporary build directory.
* `-s`, `--build-from-source`:
  Compile *`formula`* from source even if a bottle is available.
* `--force-bottle`:
  Install from a bottle if it exists for the current or newest version of macOS, even if it would not normally be used for installation.
* `--keep-tmp`:
  Retain the temporary files created during installation.
* `-f`, `--force`:
  Install without checking for previously installed keg-only or non-migrated versions.
* `-v`, `--verbose`:
  Print the verification and postinstall steps.
* `--display-times`:
  Print install times for each formula at the end of the run.

### `search` [*`options`*] [*`text`*|`/`*`text`*`/`]

Perform a substring search of cask tokens and formula names for *`text`*. If
*`text`* is flanked by slashes, it is interpreted as a regular expression. The
search for *`text`* is extended online to `homebrew/core` and `homebrew/cask`.

If no *`text`* is provided, list all locally available formulae (including tapped
ones). No online search is performed.

* `--casks`:
  List all locally available casks (including tapped ones). No online search is performed.
* `--desc`:
  Search for formulae with a description matching *`text`* and casks with a name matching *`text`*.
* `--macports`:
  Search for *`text`* in the given package manager's list.
* `--fink`:
  Search for *`text`* in the given package manager's list.
* `--opensuse`:
  Search for *`text`* in the given package manager's list.
* `--fedora`:
  Search for *`text`* in the given package manager's list.
* `--debian`:
  Search for *`text`* in the given package manager's list.
* `--ubuntu`:
  Search for *`text`* in the given package manager's list.

### `sh` [*`options`*]

Start a Homebrew build environment shell. Uses our years-battle-hardened
Homebrew build logic to help your `./configure && make && make install` or even
your `gem install` succeed. Especially handy if you run Homebrew in an
Xcode-only configuration since it adds tools like `make` to your `PATH` which
build systems would not find otherwise.

* `--env`:
  Use the standard `PATH` instead of superenv's when `std` is passed.

### `shellenv`

Print export statements. When run in a shell, this installation of Homebrew will be added to your `PATH`, `MANPATH`, and `INFOPATH`.

The variables `HOMEBREW_PREFIX`, `HOMEBREW_CELLAR` and `HOMEBREW_REPOSITORY` are also exported to avoid querying them multiple times.
Consider adding evaluation of this command's output to your dotfiles (e.g. `~/.profile` or `~/.zprofile`) with: `eval $(brew shellenv)`

### `style` [*`options`*] [*`file`*|*`tap`*|*`formula`*]

Check formulae or files for conformance to Homebrew style guidelines.

Lists of *`file`*, *`tap`* and *`formula`* may not be combined. If none are provided,
`style` will run style checks on the whole Homebrew library, including core code
and all formulae.

* `--fix`:
  Fix style violations automatically using RuboCop's auto-correct feature.
* `--display-cop-names`:
  Include the RuboCop cop name for each violation in the output.
* `--only-cops`:
  Specify a comma-separated *`cops`* list to check for violations of only the listed RuboCop cops.
* `--except-cops`:
  Specify a comma-separated *`cops`* list to skip checking for violations of the listed RuboCop cops.

### `switch` *`formula`* *`version`*

Symlink all of the specified *`version`* of *`formula`*'s installation into
Homebrew's prefix.

### `tap` [*`options`*] *`user`*`/`*`repo`* [*`URL`*]

Tap a formula repository.

If no arguments are provided, list all installed taps.

With *`URL`* unspecified, tap a formula repository from GitHub using HTTPS. Since
so many taps are hosted on GitHub, this command is a shortcut for `brew tap`
*`user`*`/`*`repo`* `https://github.com/`*`user`*`/homebrew-`*`repo`*.

With *`URL`* specified, tap a formula repository from anywhere, using any
transport protocol that `git`(1) handles. The one-argument form of `tap`
simplifies but also limits. This two-argument command makes no assumptions, so
taps can be cloned from places other than GitHub and using protocols other than
HTTPS, e.g. SSH, GIT, HTTP, FTP(S), RSYNC.

* `--full`:
  Use a full clone when tapping a repository. By default, the repository is cloned as a shallow copy (`--depth=1`). To convert a shallow copy to a full copy, you can retap by passing `--full` without first untapping.
* `--force-auto-update`:
  Auto-update tap even if it is not hosted on GitHub. By default, only taps hosted on GitHub are auto-updated (for performance reasons).
* `--repair`:
  Migrate tapped formulae from symlink-based to directory-based structure.
* `--list-pinned`:
  List all pinned taps.
* `-q`, `--quieter`:
  Suppress any warnings.

### `tap-info` [*`options`*] [*`tap`*]

Show detailed information about one or more *`tap`*s.

If no *`tap`* names are provided, display brief statistics for all installed taps.

* `--installed`:
  Show information on each installed tap.
* `--json`:
  Print a JSON representation of *`tap`*. Currently the default and only accepted value for *`version`* is `v1`. See the docs for examples of using the JSON output: <https://docs.brew.sh/Querying-Brew>

### `uninstall`, `rm`, `remove` [*`options`*] *`formula`*

Uninstall *`formula`*.

* `-f`, `--force`:
  Delete all installed versions of *`formula`*.
* `--ignore-dependencies`:
  Don't fail uninstall, even if *`formula`* is a dependency of any installed formulae.

### `unlink` [*`options`*] *`formula`*

Remove symlinks for *`formula`* from Homebrew's prefix. This can be useful for
temporarily disabling a formula: `brew unlink` *`formula`* `&&` *`commands`* `&&
brew link` *`formula`*

* `-n`, `--dry-run`:
  List files which would be unlinked without actually unlinking or deleting any files.

### `unpack` [*`options`*] *`formula`*

Unpack the source files for *`formula`* into subdirectories of the current working
directory.

* `--destdir`:
  Create subdirectories in the directory named by *`path`* instead.
* `--patch`:
  Patches for *`formula`* will be applied to the unpacked source.
* `-g`, `--git`:
  Initialise a Git repository in the unpacked source. This is useful for creating patches for the software.

### `unpin` *`formula`*

Unpin *`formula`*, allowing them to be upgraded by `brew upgrade` *`formula`*. See
also `pin`.

### `untap` *`tap`*

Remove a tapped formula repository.

### `update`, `up` [*`options`*]

Fetch the newest version of Homebrew and all formulae from GitHub using `git`(1) and perform any necessary migrations.

* `--merge`:
  Use `git merge` to apply updates (rather than `git rebase`).

### `update-reset` [*`repository`*]

Fetch and reset Homebrew and all tap repositories (or any specified *`repository`*) using `git`(1) to their latest `origin/master`.

*Note:* this will destroy all your uncommitted or committed changes.

### `upgrade` [*`options`*] [*`formula`*]

Upgrade outdated, unpinned formulae using the same options they were originally
installed with, plus any appended brew formula options. If *`formula`* are
specified, upgrade only the given *`formula`* kegs (unless they are pinned; see
`pin`, `unpin`).

Unless `HOMEBREW_NO_INSTALL_CLEANUP` is set, `brew cleanup` will then be run for
the upgraded formulae or, every 30 days, for all formulae.

* `-d`, `--debug`:
  If brewing fails, open an interactive debugging session with access to IRB or a shell inside the temporary build directory.
* `-s`, `--build-from-source`:
  Compile *`formula`* from source even if a bottle is available.
* `--force-bottle`:
  Install from a bottle if it exists for the current or newest version of macOS, even if it would not normally be used for installation.
* `--fetch-HEAD`:
  Fetch the upstream repository to detect if the HEAD installation of the formula is outdated. Otherwise, the repository's HEAD will only be checked for updates when a new stable or development version has been released.
* `--ignore-pinned`:
  Set a successful exit status even if pinned formulae are not upgraded.
* `--keep-tmp`:
  Retain the temporary files created during installation.
* `-f`, `--force`:
  Install without checking for previously installed keg-only or non-migrated versions.
* `-v`, `--verbose`:
  Print the verification and postinstall steps.
* `--display-times`:
  Print install times for each formula at the end of the run.
* `-n`, `--dry-run`:
  Show what would be upgraded, but do not actually upgrade anything.

### `uses` [*`options`*] *`formula`*

Show formulae that specify *`formula`* as a dependency. When given multiple
formula arguments, show the intersection of formulae that use *`formula`*. By
default, `uses` shows all formulae that specify *`formula`* as a required or
recommended dependency for their stable builds.

* `--recursive`:
  Resolve more than one level of dependencies.
* `--installed`:
  Only list formulae that are currently installed.
* `--include-build`:
  Include all formulae that specify *`formula`* as `:build` type dependency.
* `--include-test`:
  Include all formulae that specify *`formula`* as `:test` type dependency.
* `--include-optional`:
  Include all formulae that specify *`formula`* as `:optional` type dependency.
* `--skip-recommended`:
  Skip all formulae that specify *`formula`* as `:recommended` type dependency.
* `--devel`:
  Show usage of *`formula`* by development builds.
* `--HEAD`:
  Show usage of *`formula`* by HEAD builds.

### `--cache` [*`options`*] [*`formula`*]

Display Homebrew's download cache. See also `HOMEBREW_CACHE`.

If *`formula`* is provided, display the file or directory used to cache *`formula`*.

* `-s`, `--build-from-source`:
  Show the cache file used when building from source.
* `--force-bottle`:
  Show the cache file used when pouring a bottle.

### `--cellar` [*`formula`*]

Display Homebrew's Cellar path. *Default:* `$(brew --prefix)/Cellar`, or if that
directory doesn't exist, `$(brew --repository)/Cellar`.

If *`formula`* is provided, display the location in the cellar where *`formula`*
would be installed, without any sort of versioned directory as the last path.

### `--env` [*`options`*] [*`formula`*]

Summarise Homebrew's build environment as a plain list.

If the command's output is sent through a pipe and no shell is specified, the
list is formatted for export to `bash`(1) unless `--plain` is passed.

* `--shell`:
  Generate a list of environment variables for the specified shell, or `--shell=auto` to detect the current shell.
* `--plain`:
  Generate plain output even when piped.

### `--prefix` [*`formula`*]

Display Homebrew's install path. *Default:* `/usr/local` on macOS and
`/home/linuxbrew/.linuxbrew` on Linux.

If *`formula`* is provided, display the location in the cellar where *`formula`* is
or would be installed.

### `--repository`, `--repo` [*`user`*`/`*`repo`*]

Display where Homebrew's `.git` directory is located.

If *`user`*`/`*`repo`* are provided, display where tap *`user`*`/`*`repo`*'s directory
is located.

### `--version`

Print the version numbers of Homebrew, Homebrew/homebrew-core and
Homebrew/homebrew-cask (if tapped) to standard output.

## DEVELOPER COMMANDS

### `audit` [*`options`*] [*`formula`*]

Check *`formula`* for Homebrew coding style violations. This should be run before
submitting a new formula. If no *`formula`* are provided, check all locally
available formulae. Will exit with a non-zero status if any errors are found,
which can be useful for implementing pre-commit hooks.

* `--strict`:
  Run additional style checks, including RuboCop style checks.
* `--online`:
  Run additional slower style checks that require a network connection.
* `--new-formula`:
  Run various additional style checks to determine if a new formula is eligible for Homebrew. This should be used when creating new formula and implies `--strict` and `--online`.
* `--fix`:
  Fix style violations automatically using RuboCop's auto-correct feature.
* `--display-cop-names`:
  Include the RuboCop cop name for each violation in the output.
* `--display-filename`:
  Prefix every line of output with the file or formula name being audited, to make output easy to grep.
* `-D`, `--audit-debug`:
  Enable debugging and profiling of audit methods.
* `--only`:
  Specify a comma-separated *`method`* list to only run the methods named `audit_`*`method`*.
* `--except`:
  Specify a comma-separated *`method`* list to skip running the methods named `audit_`*`method`*.
* `--only-cops`:
  Specify a comma-separated *`cops`* list to check for violations of only the listed RuboCop cops.
* `--except-cops`:
  Specify a comma-separated *`cops`* list to skip checking for violations of the listed RuboCop cops.

### `bottle` [*`options`*] *`formula`*

Generate a bottle (binary package) from a formula that was installed with
`--build-bottle`. If the formula specifies a rebuild version, it will be
incremented in the generated DSL. Passing `--keep-old` will attempt to keep it
at its original value, while `--no-rebuild` will remove it.

* `--skip-relocation`:
  Do not check if the bottle can be marked as relocatable.
* `--force-core-tap`:
  Build a bottle even if *`formula`* is not in `homebrew/core` or any installed taps.
* `--no-rebuild`:
  If the formula specifies a rebuild version, remove it from the generated DSL.
* `--keep-old`:
  If the formula specifies a rebuild version, attempt to preserve its value in the generated DSL.
* `--json`:
  Write bottle information to a JSON file, which can be used as the value for `--merge`.
* `--merge`:
  Generate an updated bottle block for a formula and optionally merge it into the formula file. Instead of a formula name, requires the path to a JSON file generated with `brew bottle --json` *`formula`*.
* `--write`:
  Write changes to the formula file. A new commit will be generated unless `--no-commit` is passed.
* `--no-commit`:
  When passed with `--write`, a new commit will not generated after writing changes to the formula file.
* `--root-url`:
  Use the specified *`URL`* as the root of the bottle's URL instead of Homebrew's default.

### `bump-formula-pr` [*`options`*] [*`formula`*]

Create a pull request to update *`formula`* with a new URL or a new tag.

If a *`URL`* is specified, the *`SHA-256`* checksum of the new download should also
be specified. A best effort to determine the *`SHA-256`* and *`formula`* name will
be made if either or both values are not supplied by the user.

If a *`tag`* is specified, the Git commit *`revision`* corresponding to that tag
must also be specified.

*Note:* this command cannot be used to transition a formula from a
URL-and-SHA-256 style specification into a tag-and-revision style specification,
nor vice versa. It must use whichever style specification the formula already
uses.

* `--devel`:
  Bump the development rather than stable version. The development spec must already exist.
* `-n`, `--dry-run`:
  Print what would be done rather than doing it.
* `--write`:
  When passed along with `--dry-run`, perform a not-so-dry run by making the expected file modifications but not taking any Git actions.
* `--no-audit`:
  Don't run `brew audit` before opening the PR.
* `--strict`:
  Run `brew audit --strict` before opening the PR.
* `--no-browse`:
  Print the pull request URL instead of opening in a browser.
* `--no-fork`:
  Don't try to fork the repository.
* `--mirror`:
  Use the specified *`URL`* as a mirror URL.
* `--version`:
  Use the specified *`version`* to override the value parsed from the URL or tag. Note that `--version=0` can be used to delete an existing version override from a formula if it has become redundant.
* `--message`:
  Append *`message`* to the default pull request message.
* `--url`:
  Specify the *`URL`* for the new download. If a *`URL`* is specified, the *`SHA-256`* checksum of the new download should also be specified.
* `--sha256`:
  Specify the *`SHA-256`* checksum of the new download.
* `--tag`:
  Specify the new git commit *`tag`* for the formula.
* `--revision`:
  Specify the new git commit *`revision`* corresponding to the specified *`tag`*.

### `bump-revision` [*`options`*] *`formula`*

Create a commit to increment the revision of *`formula`*. If no revision is
present, "revision 1" will be added.

* `-n`, `--dry-run`:
  Print what would be done rather than doing it.
* `--message`:
  Append *`message`* to the default commit message.

### `create` [*`options`*] *`URL`*

Generate a formula for the downloadable file at *`URL`* and open it in the editor.
Homebrew will attempt to automatically derive the formula name and version, but
if it fails, you'll have to make your own template. The `wget` formula serves as
a simple example. For the complete API, see:
<http://www.rubydoc.info/github/Homebrew/brew/master/Formula>

* `--autotools`:
  Create a basic template for an Autotools-style build.
* `--cmake`:
  Create a basic template for a CMake-style build.
* `--go`:
  Create a basic template for a Go build.
* `--meson`:
  Create a basic template for a Meson-style build.
* `--perl`:
  Create a basic template for a Perl build.
* `--python`:
  Create a basic template for a Python build.
* `--rust`:
  Create a basic template for a Rust build.
* `--no-fetch`:
  Homebrew will not download *`URL`* to the cache and will thus not add its SHA-256 to the formula for you, nor will it check the GitHub API for GitHub projects (to fill out its description and homepage).
* `--HEAD`:
  Indicate that *`URL`* points to the package's repository rather than a file.
* `--set-name`:
  Explicitly set the *`name`* of the new formula.
* `--set-version`:
  Explicitly set the *`version`* of the new formula.
* `--tap`:
  Generate the new formula within the given tap, specified as *`user`*`/`*`repo`*.

### `edit` [*`formula`*]

Open *`formula`* in the editor set by `EDITOR` or `HOMEBREW_EDITOR`, or open the
Homebrew repository for editing if no formula is provided.

### `extract` [*`options`*] *`formula`* *`tap`*

Look through repository history to find the most recent version of *`formula`* and
create a copy in *`tap`*`/Formula/`*`formula`*`@`*`version`*`.rb`. If the tap is not
installed yet, attempt to install/clone the tap before continuing. To extract a
formula from a tap that is not `homebrew/core` use its fully-qualified form of
*`user`*`/`*`repo`*`/`*`formula`*.

* `--version`:
  Extract the specified *`version`* of *`formula`* instead of the most recent.

### `formula` *`formula`*

Display the path where *`formula`* is located.

### `install-bundler-gems`

Install Homebrew's Bundler gems.

### `irb` [*`options`*]

Enter the interactive Homebrew Ruby shell.

* `--examples`:
  Show several examples.
* `--pry`:
  Use Pry instead of IRB. Implied if `HOMEBREW_PRY` is set.

### `linkage` [*`options`*] [*`formula`*]

Check the library links from the given *`formula`* kegs. If no *`formula`* are
provided, check all kegs. Raises an error if run on uninstalled formulae.

* `--test`:
  Show only missing libraries and exit with a non-zero status if any missing libraries are found.
* `--reverse`:
  For every library that a keg references, print its dylib path followed by the binaries that link to it.
* `--cached`:
  Print the cached linkage values stored in `HOMEBREW_CACHE`, set by a previous `brew linkage` run.

### `man` [*`options`*]

Generate Homebrew's manpages.

* `--fail-if-changed`:
  Return a failing status code if changes are detected in the manpage outputs. This can be used to notify CI when the manpages are out of date. Additionally, the date used in new manpages will match those in the existing manpages (to allow comparison without factoring in the date).
* `--link`:
  This is now done automatically by `brew update`.

### `prof` *`command`*

Run Homebrew with the Ruby profiler, e.g. `brew prof readall`.

### `pull` [*`options`*] *`patch`*

Get a patch from a GitHub commit or pull request and apply it to Homebrew.
Optionally, publish updated bottles for any formulae changed by the patch.

Each *`patch`* may be the number of a pull request in `homebrew/core`, the URL of
any pull request or commit on GitHub or a "https://jenkins.brew.sh/job/..."
testing job URL.

* `--bottle`:
  Handle bottles, pulling the bottle-update commit and publishing files on Bintray.
* `--bump`:
  For one-formula PRs, automatically reword commit message to our preferred format.
* `--clean`:
  Do not rewrite or otherwise modify the commits found in the pulled PR.
* `--ignore-whitespace`:
  Silently ignore whitespace discrepancies when applying diffs.
* `--resolve`:
  When a patch fails to apply, leave in progress and allow user to resolve, instead of aborting.
* `--branch-okay`:
  Do not warn if pulling to a branch besides master (useful for testing).
* `--no-pbcopy`:
  Do not copy anything to the system clipboard.
* `--no-publish`:
  Do not publish bottles to Bintray.
* `--warn-on-publish-failure`:
  Do not exit if there's a failure publishing bottles on Bintray.
* `--bintray-org`:
  Publish bottles to the specified Bintray *`organisation`*.
* `--test-bot-user`:
  Pull the bottle block commit from the specified *`user`* on GitHub.

### `release-notes` [*`options`*] [*`previous_tag`*] [*`end_ref`*]

Print the merged pull requests on Homebrew/brew between two Git refs. If no
*`previous_tag`* is provided it defaults to the latest tag. If no *`end_ref`* is
provided it defaults to `origin/master`.

* `--markdown`:
  Print as a Markdown list.

### `ruby` (`-e` *`text`*|*`file`*)

Run a Ruby instance with Homebrew's libraries loaded, e.g. `brew ruby -e "puts
:gcc.f.deps"` or `brew ruby script.rb`.

* `-r`:
  Load a library using `require`.
* `-e`:
  Execute the given text string as a script.

### `tap-new` *`user`*`/`*`repo`*

Generate the template files for a new tap.

### `test` [*`options`*] *`formula`*

Run the test method provided by an installed formula. There is no standard
output or return code, but generally it should notify the user if something is
wrong with the installed formula.

*Example:* `brew install jruby && brew test jruby`

* `--devel`:
  Test the development version of a formula.
* `--HEAD`:
  Test the head version of a formula.
* `--keep-tmp`:
  Retain the temporary files created for the test.

### `tests` [*`options`*]

Run Homebrew's unit and integration tests.

* `--coverage`:
  Generate code coverage reports.
* `--generic`:
  Run only OS-agnostic tests.
* `--no-compat`:
  Do not load the compatibility layer when running tests.
* `--online`:
  Include tests that use the GitHub API and tests that use any of the taps for official external commands.
* `--only`:
  Run only *`test_script`*`_spec.rb`. Appending `:`*`line_number`* will start at a specific line.
* `--seed`:
  Randomise tests with the specified *`value`* instead of a random seed.

### `update-test` [*`options`*]

Run a test of `brew update` with a new repository clone. If no options are
passed, use `origin/master` as the start commit.

* `--to-tag`:
  Set `HOMEBREW_UPDATE_TO_TAG` to test updating between tags.
* `--keep-tmp`:
  Retain the temporary directory containing the new repository clone.
* `--commit`:
  Use the specified *`commit`* as the start commit.
* `--before`:
  Use the commit at the specified *`date`* as the start commit.

### `vendor-gems`

Install and commit Homebrew's vendored gems.

## GLOBAL OPTIONS

These options are applicable across all sub-commands.

* `-q`, `--quiet`:
  Suppress any warnings.

* `-v`, `--verbose`:
  Make some output more verbose.

* `-d`, `--debug`:
  Display any debugging information.

* `-f`, `--force`:
  Override warnings and enable potentially unsafe operations.

## OFFICIAL EXTERNAL COMMANDS

### `bundle` *`subcommand`*:

Bundler for non-Ruby dependencies from Homebrew, Homebrew Cask and the Mac App Store.
See `brew bundle --help`.

**Homebrew/homebrew-bundle**: <https://github.com/Homebrew/homebrew-bundle>

### `cask` *`subcommand`*:

Install macOS applications distributed as binaries. See `brew-cask`(1).

**Homebrew/homebrew-cask**: <https://github.com/Homebrew/homebrew-cask>

### `services` *`subcommand`*:

Manage background services with macOS' `launchctl`(1) daemon manager.
See `brew services --help`.

**Homebrew/homebrew-services**: <https://github.com/Homebrew/homebrew-services>

## CUSTOM EXTERNAL COMMANDS

Homebrew, like `git`(1), supports external commands. These are executable
scripts that reside somewhere in the `PATH`, named `brew-`*`cmdname`* or
`brew-`*`cmdname`*`.rb`, which can be invoked like `brew` *`cmdname`*. This allows you
to create your own commands without modifying Homebrew's internals.

Instructions for creating your own commands can be found in the docs:
<https://docs.brew.sh/External-Commands>

## SPECIFYING FORMULAE

Many Homebrew commands accept one or more *`formula`* arguments. These arguments
can take several different forms:

  * The name of a formula:
    e.g. `git`, `node`, `wget`.

  * The fully-qualified name of a tapped formula:
    Sometimes a formula from a tapped repository may conflict with one in
    `homebrew/core`.
    You can still access these formulae by using a special syntax, e.g.
    `homebrew/dupes/vim` or `homebrew/versions/node4`.

  * An arbitrary file or URL:
    Homebrew can install formulae via URL, e.g.
    `https://raw.githubusercontent.com/Homebrew/homebrew-core/master/Formula/git.rb`,
    or from a local path. It can point to either a formula file or a bottle.
    In the case of a URL, the downloaded file will be cached for later use.

## ENVIRONMENT

Note that environment variables must have a value set to be detected. For example, run
`export HOMEBREW_NO_INSECURE_REDIRECT=1` rather than just
`export HOMEBREW_NO_INSECURE_REDIRECT`.

  * `HOMEBREW_ARCH`:
    Linux only: If set, Homebrew will pass the set value to type name to the compiler's
    `-march` option instead of using the default (`-march=native`).

  * `HOMEBREW_ARTIFACT_DOMAIN`:
    If set, instructs Homebrew to prefix all download URLs, including those for bottles,
    with this variable. For example, `HOMEBREW_ARTIFACT_DOMAIN=http://localhost:8080`
    will cause a formula with the URL `https://example.com/foo.tar.gz` to instead
    download from `http://localhost:8080/example.com/foo.tar.gz`.

  * `HOMEBREW_AUTO_UPDATE_SECS`:
    If set, Homebrew will only check for autoupdates once per this seconds interval.

    *Default:* `300`.

  * `HOMEBREW_AWS_ACCESS_KEY_ID`, `HOMEBREW_AWS_SECRET_ACCESS_KEY`:
    When using the `S3` download strategy, Homebrew will look in
    these variables for access credentials (see
    <https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-getting-started.html#cli-environment>
    to retrieve these access credentials from AWS). If they are not set,
    the `S3` download strategy will download with a public (unsigned) URL.

  * `HOMEBREW_BAT`:
    If set, Homebrew will use `bat` for the `brew cat` command.

  * `HOMEBREW_BOTTLE_DOMAIN`:
    By default, Homebrew uses `https://homebrew.bintray.com/` as its download
    mirror for bottles. If set, instructs Homebrew to instead use the specified
    URL. For example, `HOMEBREW_BOTTLE_DOMAIN=http://localhost:8080` will
    cause all bottles to download from the prefix `http://localhost:8080/`.

  * `HOMEBREW_BREW_GIT_REMOTE`:
    By default, Homebrew uses `https://github.com/Homebrew/brew` as its
    Homebrew/brew `git`(1) remote. If set, instructs Homebrew to instead use
    the specified URL.

  * `HOMEBREW_BROWSER`:
    If set, Homebrew uses this setting as the browser when opening project
    homepages, instead of the OS default browser.

  * `HOMEBREW_CACHE`:
    If set, instructs Homebrew to use the specified directory as the download cache.

    *Default:* `~/Library/Caches/Homebrew`.

  * `HOMEBREW_CORE_GIT_REMOTE`:
    By default, Homebrew uses `https://github.com/Homebrew/homebrew-core` (or
    `https://github.com/Homebrew/linuxbrew-core`) as its Homebrew/homebrew-core
    `git`(1) remote. If set, instructs Homebrew to instead use the specified URL.

  * `HOMEBREW_CURLRC`:
    If set, Homebrew will not pass `-q` when invoking `curl`(1), which disables
    the use of `curlrc`.

  * `HOMEBREW_CURL_VERBOSE`:
    If set, Homebrew will pass `--verbose` when invoking `curl`(1).

  * `HOMEBREW_CURL_RETRIES`:
    If set, Homebrew will pass the given retry count to `--retry` when invoking `curl`(1).

  * `HOMEBREW_DEBUG`:
    If set, any commands that can emit debugging information will do so.

  * `HOMEBREW_DEVELOPER`:
    If set, Homebrew will tweak behaviour to be more relevant for Homebrew
    developers (active or budding), e.g. turning warnings into errors.

  * `HOMEBREW_DISPLAY`:
    If set, Homebrew will use this X11 display when opening a page in a browser,
    for example with `brew home`. Primarily useful on Linux.

    *Default:* the value of the user's `DISPLAY` environment variable.

  * `HOMEBREW_DISPLAY_INSTALL_TIMES`:
    If set, Homebrew will print install times for each formula at the end of the run.

  * `HOMEBREW_EDITOR`:
    If set, Homebrew will use this editor when editing a single formula, or
    several formulae in the same directory.

    *Note:* `brew edit` will open all of Homebrew as discontinuous files and
    directories. TextMate can handle this correctly in project mode, but many
    editors will do strange things in this case.

  * `HOMEBREW_FORCE_BREWED_CURL`:
    If set, Homebrew will always use a Homebrew-installed `curl` rather than the
    system version. Automatically set if the system version of `curl` is too old.

  * `HOMEBREW_FORCE_VENDOR_RUBY`:
    If set, Homebrew will always use its vendored, relocatable Ruby version
    even if the system version of Ruby is new enough.

  * `HOMEBREW_FORCE_BREWED_GIT`:
    If set, Homebrew will always use a Homebrew-installed `git`(1) rather than the
    system version. Automatically set if the system version of `git` is too old.

  * `HOMEBREW_GITHUB_API_TOKEN`:
    A personal access token for the GitHub API, used by Homebrew for features
    such as `brew search`. You can create one at <https://github.com/settings/tokens>.
    If set, GitHub will allow you a greater number of API requests. For more
    information, see: <https://developer.github.com/v3/#rate-limiting>

    *Note:* Homebrew doesn't require permissions for any of the scopes.

  * `HOMEBREW_INSTALL_BADGE`:
    Text printed before the installation summary of each successful build.

    *Default:* the beer emoji.

  * `HOMEBREW_LOGS`:
    If set, Homebrew will use the specified directory to store log files.

  * `HOMEBREW_MAKE_JOBS`:
    If set, instructs Homebrew to use the value of `HOMEBREW_MAKE_JOBS` as
    the number of parallel jobs to run when building with `make`(1).

    *Default:* the number of available CPU cores.

  * `HOMEBREW_NO_ANALYTICS`:
    If set, Homebrew will not send analytics. See: <https://docs.brew.sh/Analytics>

  * `HOMEBREW_NO_AUTO_UPDATE`:
    If set, Homebrew will not auto-update before running `brew install`,
    `brew upgrade` or `brew tap`.

  * `HOMEBREW_NO_BOTTLE_SOURCE_FALLBACK`:
    If set, Homebrew will fail on the failure of installation from a bottle
    rather than falling back to building from source.

  * `HOMEBREW_NO_COLOR`:
    If set, Homebrew will not print text with colour added.

  * `HOMEBREW_NO_EMOJI`:
    If set, Homebrew will not print the `HOMEBREW_INSTALL_BADGE` on a
    successful build.

    *Note:* Homebrew will only try to print emoji on OS X Lion or newer.

  * `HOMEBREW_NO_INSECURE_REDIRECT`:
    If set, Homebrew will not permit redirects from secure HTTPS
    to insecure HTTP.

    While ensuring your downloads are fully secure, this is likely
    to cause from-source SourceForge, some GNU & GNOME based
    formulae to fail to download.

  * `HOMEBREW_NO_GITHUB_API`:
    If set, Homebrew will not use the GitHub API, e.g. for searches or
    fetching relevant issues on a failed install.

  * `HOMEBREW_NO_INSTALL_CLEANUP`:
    If set, `brew install`, `brew upgrade` and `brew reinstall` will never
    automatically cleanup the installed/upgraded/reinstalled formulae or all
    formulae every 30 days.

  * `HOMEBREW_PRY`:
    If set, Homebrew will use Pry for the `brew irb` command.

  * `HOMEBREW_SVN`:
    When exporting from Subversion, Homebrew will use `HOMEBREW_SVN` if set,
    a Homebrew-built Subversion if installed, or the system-provided binary.

    Set this to force Homebrew to use a particular `svn` binary.

  * `HOMEBREW_TEMP`:
    If set, instructs Homebrew to use `HOMEBREW_TEMP` as the temporary directory
    for building packages. This may be needed if your system temp directory and
    Homebrew prefix are on different volumes, as macOS has trouble moving
    symlinks across volumes when the target does not yet exist.

    This issue typically occurs when using FileVault or custom SSD configurations.

  * `HOMEBREW_UPDATE_TO_TAG`:
    If set, instructs Homebrew to always use the latest stable tag (even if
    developer commands have been run).

  * `HOMEBREW_VERBOSE`:
    If set, Homebrew always assumes `--verbose` when running commands.

  * `http_proxy`:
    Sets the HTTP proxy to be used by `curl`, `git` and `svn` when downloading
    through Homebrew.

  * `https_proxy`:
    Sets the HTTPS proxy to be used by `curl`, `git` and `svn` when downloading
    through Homebrew.

  * `all_proxy`:
    Sets the SOCKS5 proxy to be used by `curl`, `git` and `svn` when downloading
    through Homebrew.

  * `ftp_proxy`:
    Sets the FTP proxy to be used by `curl`, `git` and `svn` when downloading
    through Homebrew.

  * `no_proxy`:
    Sets the comma-separated list of hostnames and domain names that should be excluded
    from proxying by `curl`, `git` and `svn` when downloading through Homebrew.

## USING HOMEBREW BEHIND A PROXY

Set the `http_proxy`, `https_proxy`, `all_proxy`, `ftp_proxy` and/or `no_proxy`
environment variables documented above.

For example, to use an unauthenticated HTTP or SOCKS5 proxy:

    export http_proxy=http://$HOST:$PORT

    export all_proxy=socks5://$HOST:$PORT

And for an authenticated HTTP proxy:

    export http_proxy=http://$USER:$PASSWORD@$HOST:$PORT

## SEE ALSO

Homebrew Documentation: <https://docs.brew.sh>

Homebrew API: <https://rubydoc.brew.sh>

`brew-cask`(1), `git`(1), `git-log`(1)

## AUTHORS

Homebrew's Project Leader is Mike McQuaid.

Homebrew's Project Leadership Committee is Misty De Meo, Shaun Jackman, Jonathan Chang, Mike McQuaid and Markus Reiter.

Homebrew's Technical Steering Committee is Michka Popoff, FX Coudert, Markus Reiter, Misty De Meo and Mike McQuaid.

Homebrew/brew's Linux maintainers are Michka Popoff, Shaun Jackman, Dawid Dziurla and Issy Long.

Homebrew's other current maintainers are Claudia Pellegrino, Rui Chen, Vitor Galvao, Caleb Xu, Gautham Goli, Steven Peters, William Woodruff, Igor Kapkov, Izaak Beekman, Sean Molenaar, Jan Viljanen, Eric Knibbe, Viktor Szakats, Thierry Moisan, Steven Peters, Tom Schoonjans and Issy Long.

Former maintainers with significant contributions include JCount, commitay, Dominyk Tiller, Tim Smith, Baptiste Fontaine, Xu Cheng, Martin Afanasjew, Brett Koonce, Charlie Sharpsteen, Jack Nagel, Adam Vandenberg, Andrew Janke, Alex Dunn, neutric, Tomasz Pajor, Uladzislau Shablinski, Alyssa Ross, ilovezfs, Chongyu Zhu and Homebrew's creator: Max Howell.

## BUGS

See our issues on GitHub:

  * **Homebrew/brew**:
    <https://github.com/Homebrew/brew/issues>

  * **Homebrew/homebrew-core**:
    <https://github.com/Homebrew/homebrew-core/issues>

[SYNOPSIS]: #SYNOPSIS "SYNOPSIS"
[DESCRIPTION]: #DESCRIPTION "DESCRIPTION"
[ESSENTIAL COMMANDS]: #ESSENTIAL-COMMANDS "ESSENTIAL COMMANDS"
[COMMANDS]: #COMMANDS "COMMANDS"
[DEVELOPER COMMANDS]: #DEVELOPER-COMMANDS "DEVELOPER COMMANDS"
[GLOBAL OPTIONS]: #GLOBAL-OPTIONS "GLOBAL OPTIONS"
[OFFICIAL EXTERNAL COMMANDS]: #OFFICIAL-EXTERNAL-COMMANDS "OFFICIAL EXTERNAL COMMANDS"
[CUSTOM EXTERNAL COMMANDS]: #CUSTOM-EXTERNAL-COMMANDS "CUSTOM EXTERNAL COMMANDS"
[SPECIFYING FORMULAE]: #SPECIFYING-FORMULAE "SPECIFYING FORMULAE"
[ENVIRONMENT]: #ENVIRONMENT "ENVIRONMENT"
[USING HOMEBREW BEHIND A PROXY]: #USING-HOMEBREW-BEHIND-A-PROXY "USING HOMEBREW BEHIND A PROXY"
[SEE ALSO]: #SEE-ALSO "SEE ALSO"
[AUTHORS]: #AUTHORS "AUTHORS"
[BUGS]: #BUGS "BUGS"

[-]: -.html
