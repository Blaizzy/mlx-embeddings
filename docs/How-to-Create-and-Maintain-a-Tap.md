# How to Create and Maintain a Tap

[Taps](Taps.md) are external sources of Homebrew formulae, casks  and/or external commands. They
can be created by anyone to provide their own formulae, casks  and/or external commands
to any Homebrew user.

## Creating a tap

A tap is usually a Git repository available online, but you can use anything as
long as it’s a protocol that Git understands, or even just a directory with
files in it.
If hosted on GitHub, we recommend that the repository’s name start with
`homebrew-` so the short `brew tap` command can be used.
See the [manpage](Manpage.md) for more information on repository naming.

The `brew tap-new` command can be used to create a new tap along with some
template files.

Tap formulae follow the same format as the core’s ones, and can be added at the
repository’s root, or under `Formula` or `HomebrewFormula` subdirectories. We
recommend the latter options because it makes the repository organisation
easier to grasp, and top-level files are not mixed with formulae.

See [homebrew/core](https://github.com/Homebrew/homebrew-core) for an example of
a tap with a `Formula` subdirectory.

## Naming your formulae to avoid clashes

If your formulae have the same name as Homebrew/homebrew-core formulae they cannot be installed side-by-side. If you wish to create a different version of a formula that's in Homebrew/homebrew-core (e.g. with `option`s) consider giving it a different name e.g. `nginx-full` for more fully-featured `nginx` formula. This will allow both `nginx` and `nginx-full` to be installed at the same time (but not linked if there are conflicts and one of them is not declared to be `keg_only`).

### Installing

If it’s on GitHub, users can install any of your formulae with
`brew install user/repo/formula`. Homebrew will automatically add your
`github.com/user/homebrew-repo` tap before installing the formula.
`user/repo/formula` points to the `github.com/user/homebrew-repo/**/formula.rb`
file here.

If they want to get your tap without installing any formula at the same time,
users can add it with the [`brew tap` command](Taps.md).

If it’s on GitHub, they can use `brew tap user/repo`, where `user` is your
GitHub username and `homebrew-repo` is your repository.

If it’s hosted outside of GitHub, they have to use `brew tap user/repo <URL>`,
where `user` and `repo` will be used to refer to your tap and `<URL>` is your
Git clone URL.

Users can then install your formulae either with `brew install foo` if there’s
no core formula with the same name, or with `brew install user/repo/foo` to
avoid conflicts.

## Maintaining a tap

A tap is just a Git repository so you don’t have to do anything specific when
making modifications, apart from committing and pushing your changes.

### Updating

Once your tap is installed, Homebrew will update it each time a user runs
`brew update`. Outdated formulae will be upgraded when a user runs
`brew upgrade`, like core formulae.

## Casks

Casks can also be installed from a tap.
Casks can be included in taps with formulae, or in a tap with just casks.
Place any cask files you wish to make available in a `Casks` directory at the top level of your tap.

See [homebrew/cask](https://github.com/Homebrew/homebrew-cask) for an example of a tap with a `Casks` subdirectory.

### Naming

Unlike formulae, casks must have globally unique names to avoid clashes.
This can be achieved by e.g. prepending the cask name with you github username: `username-formula-name`.

## External commands

You can provide your tap users with custom `brew` commands by adding them in a
`cmd` subdirectory. [Read more on external commands](External-Commands.md).

See [homebrew/aliases](https://github.com/Homebrew/homebrew-aliases) for an
example of a tap with external commands.
