# Homebrew/linuxbrew-core Maintainer Guide

## Merging formulae updates from Homebrew/homebrew-core

Linuxbrew-core is a fork of Homebrew-core and, therefore, it has to periodically
merge changes made by Homebrew developers and contributors. Below we
describe the steps required to merge `Homebrew/homebrew-core` into
`Linuxbrew/homebrew-core`, possible conflicts and ways to resolve
them. Note, that instructions below have been written for a "clean"
environment and you might be able to skip some of the steps if you
have done them in the past.

### Preparation

First of all, we want to enable developer commands and prevent
automatic updates while we do the merge:

```bash
export HOMEBREW_DEVELOPER=1
export HOMEBREW_NO_AUTO_UPDATE=1
```

Once we've done that, we need to get access to the `merge-homebrew`
command that will be used for the merge. To do that we have to tap the
[`Homebrew/linux-dev`](https://github.com/Homebrew/homebrew-linux-dev)
repository:

```bash
brew tap homebrew/linux-dev
```

Next, we have to navigate to the repository where we want to do the
merge and make sure that there are 3 remotes:

* a remote named `origin` pointing to Linuxbrew-core,
* a remote named `homebrew` pointing to Homebrew-core, and
* a remote pointing to your GitHub fork of Linuxbrew-core.

Remote names `origin` and `homebrew` are hard-coded in
`merge-homebrew`, while the remote pointing to your fork must be the
same as your GitHub username, as it will be used to submit a pull
request for the merge. Set the name to the `$HOMEBREW_GITHUB_USER` environment
variable, or let `hub fork` add a remote for you.

```bash
brew install hub
cd $(brew --repo homebrew/core)
git remote add homebrew https://github.com/Homebrew/homebrew-core.git
hub fork --remote-name=$HOMEBREW_GITHUB_USER
```

Now, let's make sure that our local branch `master` is clean and that
your fork is up-to-date with Homebrew/linuxbrew-core:

```bash
git checkout master
git fetch origin master
git reset --hard origin/master
git push --force $HOMEBREW_GITHUB_USER master
```

Strictly speaking, there is no need for `git reset --hard
origin/master` and simple `git merge origin/master` would have been
sufficient if you didn't mess with your local `master` branch.
However, hard reset makes sure that these instructions are correct
even if you did mess something up. The same is true for the `--force`
flag for the `git push` command above.

By default, the following command will attempt to merge all the
changes that the upstream Homebrew developers have made.

```bash
brew merge-homebrew --core
```

Merging all the changes from upstream in one go can make it
harder to keep track of all the active builds. Instead, attempt
to only merge 8-10 modified formulae.

`git log --oneline master..homebrew/master` will show a list of all
the upstream commits since the last merge, from oldest to newest.

Pick a commit SHA-1 that will merge between 8-10 formulae (16-20 commits
including bottles). Once you're satisfied with the list of updated
formulae, begin the merge:

```bash
brew merge-homebrew --core <sha>
```

#### Simple Conflicts

Once you issue the above command, the merge will begin and in the very
end you will see the list of (conflicting) formulae that
`merge-homebrew` could not merge automatically:

```bash
==> Conflicts
Formula/git-lfs.rb Formula/gnutls.rb Formula/godep.rb
```

Note, that you can also get a list of unmerged files (*i.e.* files with conflicts) using:

```sh
git diff --name-only --diff-filter=U
```

Of course, conflicts will be different every merge. You have to
resolve these conflicts either manually in a text editor, or by using
tools like `diffuse`, `tkdiff`, or `meld`, some of which are available
from Homebrew. Frequently, conflicts are caused by the new versions
of macOS bottles and look like:

```ruby
<<<<<<< HEAD
    sha256 "bd66be269cbfe387920651c5f4f4bc01e0793034d08b5975f35f7fdfdb6c61a7" => :sierra
    sha256 "7071cb98f72c73adb30afbe049beaf947fabfeb55e9f03e0db594c568d77d69d" => :el_capitan
    sha256 "c7c0fe2464771bdcfd626fcbda9f55cb003ac1de060c51459366907edd912683" => :yosemite
    sha256 "95d4c82d38262a4bc7ef4f0a10ce2ecf90e137b67df15f8bf8df76e962e218b6" => :x86_64_linux
=======
    sha256 "ee6db42174fdc572d743e0142818b542291ca2e6ea3c20ff6a47686589cdc274" => :sierra
    sha256 "e079a92a6156e2c87c59a59887d0ae0b6450d6f3a9c1fe14838b6bc657faefaa" => :el_capitan
    sha256 "c334f91d5809d2be3982f511a3dfe9a887ef911b88b25f870558d5c7e18a15ad" => :yosemite
>>>>>>> homebrew/master
```

For such conflicts, simply remove the "HEAD" (Linuxbrew's) part of the
conflict along with `<<<<<<< HEAD`, `=======`, and `>>>>>>>
homebrew/master` lines. Later, we will submit a request to rebuild
bottles for Linux for such formulae.

The `merge-homebrew` script will stage resolved conflicts for you.

#### Complex Conflicts

Of course, from time to time conflicts are more complicated and you
have to look carefully into what's going on. An example of a slightly
more complex conflict is below:

```ruby
<<<<<<< HEAD
    if OS.mac?
      lib.install "out-shared/libleveldb.dylib.1.19" => "libleveldb.1.19.dylib"
      lib.install_symlink lib/"libleveldb.1.19.dylib" => "libleveldb.dylib"
      lib.install_symlink lib/"libleveldb.1.19.dylib" => "libleveldb.1.dylib"
      system "install_name_tool", "-id", "#{lib}/libleveldb.1.dylib", "#{lib}/libleveldb.1.19.dylib"
    else
      lib.install Dir["out-shared/libleveldb.so*"]
    end
=======
    lib.install "out-shared/libleveldb.dylib.1.19" => "libleveldb.1.19.dylib"
    lib.install_symlink lib/"libleveldb.1.19.dylib" => "libleveldb.dylib"
    lib.install_symlink lib/"libleveldb.1.19.dylib" => "libleveldb.1.dylib"
    MachO::Tools.change_dylib_id("#{lib}/libleveldb.1.dylib", "#{lib}/libleveldb.1.19.dylib")
>>>>>>> homebrew/master
```

Note, that in the "HEAD" (Linuxbrew's) part we see previous code of
the Homebrew's formula wrapped in `if OS.mac?`. To resolve such a
conflict you have to replace the contents of `if OS.mac?` part up
until `else` with the contents of the bottom part of the conflict
("homebrew/master"). You also have to check if there are any obvious
modifications that have to be made to the `else` part of the code that
deals with non-macOS-related code.


#### Finishing the merge

Once all the conflicts have been resolved, a text editor will open
with pre-populated commit message title and body:

```text
Merge branch homebrew/master into linuxbrew/master

# Conflicts:
#       Formula/git-lfs.rb
#       Formula/gnutls.rb
#       Formula/godep.rb
```

Leave the title of the message unchanged and uncomment all the
conflicting files. Your final commit message should be:

```text
Merge branch homebrew/master into linuxbrew/master

Conflicts:
        Formula/git-lfs.rb
        Formula/gnutls.rb
        Formula/godep.rb
```

#### Submitting a PR

The `merge-homebrew` command will create a pull-request for you, using `hub`.

Continuous integration verifies that the pull request passes
`brew readall` and `brew style`, which only takes a few minutes.
Finalise the merge with:

```bash
git push origin master
```

If the above command fails (e.g. another maintainer pushed changes to
Homebrew/linuxbrew-core before you finished the merge),
you can update your branch with `git rebase --rebase-merges`,
but it's often easier to just run `git reset --hard origin/master`
and redo `brew merge-homebrew --core`.

Otherwise, the merge is now complete. Don't forget to update your GitHub
fork by running `git push your-fork master`.

## Building bottles for updated formulae

After merging changes, we must rebuild bottles for all the PRs that
had conflicts. There is an automatic workflow job that handles this
when the merge commit is pushed to the repository; however, to do it
manually, tap `Homebrew/homebrew-linux-dev` and run the following
command where the merge commit is `HEAD`:

```sh
for formula in $(brew find-formulae-to-bottle); do
  brew request-bottle $formula
done
```

The `find-formulae-to-bottle` command outputs a list of formulae
parsed from the merge commit body. It also performs some checks
against the formulae:

And it skips formulae if any of the following are true:
- it doesn't need a bottle
- it already has a bottle
- the formula's tap is Homebrew/homebrew-core (the upstream macOS repository)
- there is already an open PR for the formula's bottle
- the current branch is not master

If a formula you are expecting to bottle is skipped, there may be an
error; by default, this script won't output the errors. To see them,
run `brew find-formulae-to-bottle --verbose` separate to the `for`
loop above.

The `request-bottle` script kicks off a GitHub Action to build the
bottle. If successful, it pushes the bottle to BinTray and a commit
with the SHA to `master`. There are no pull requests, and no manual
steps unless the formula fails to build. Check that the build was
successful from the [Actions tab](https://github.com/homebrew/linuxbrew-core/actions).

If the formula fails to build, we open a pull request with the fix,
which will build (but not publish) bottles for that formula. Once the
formula builds correctly, we merge that pull request from the GitHub
web interface, which starts a workflow job to publish the bottle and
push a bottle commit to Homebrew/linuxbrew-core.

## Creating new Linux-specific formula

Make a PR to `Homebrew/linuxbrew-core` containing one commit named
like this: `name (new formula)`. Keep only one commit in this PR,
squash and force push to your branch if needed. Include the line
`depends_on :linux` in the dependencies section, so that maintainers
can easily find Linux-only formulae.

For the bottle commit to be successful when new formulae are added, we
have to insert an empty bottle block into the formula code. This
usually goes after the `url` and `sha256` lines, with a blank line in
between.

```ruby
bottle do
end
```

## Common build failures and how to handle them

### Tests errors

#### "undefined reference to ..."

This error might pop up when parameters passed to `gcc` are in the wrong order.

An example from `libmagic` formula:

```
==> brew test libmagic --verbose
Testing libmagic
==> /usr/bin/gcc -I/home/linuxbrew/.linuxbrew/Cellar/libmagic/5.38/include -L/home/linuxbrew/.linuxbrew/Cellar/libmagic/5.38/lib -lmagic test.c -o test
/tmp/ccNeDVRt.o: In function `main':
test.c:(.text+0x15): undefined reference to `magic_open'
test.c:(.text+0x4a): undefined reference to `magic_load'
test.c:(.text+0x81): undefined reference to `magic_file'
collect2: error: ld returned 1 exit status
```

Solution:

```ruby
if OS.mac?
    system ENV.cc, "-I#{include}", "-L#{lib}", "-lmagic", "test.c", "-o", "test"
else
    system ENV.cc, "test.c", "-I#{include}", "-L#{lib}", "-lmagic", "-o", "test"
end
```

For an explanation of why this happens, read the [Ubuntu 11.04 Toolchain documentation](https://wiki.ubuntu.com/NattyNarwhal/ToolchainTransition).

### Bottling errors

#### Wrong cellar line

This situation might happen after merging `homebrew-core` to `linuxbrew-core`.
Linux and macOS cellars may differ and we must correct it to the value suggested by `brew`.

Example:

```
Error: --keep-old was passed but there are changes in:
cellar: old: :any, new: "/home/linuxbrew/.linuxbrew/Cellar"
==> FAILED
```

In this case deleting `cellar :any` line from the `bottle do` block is enough.

There are some formulae that would fail with an error message like the one provided above, but they are crucial for users of old systems and we should restore the `cellar :any` line after pulling the bottles.
Those formulae are:
- `patchelf`
- `binutils`
- `gcc`
- `curl`

Setting `cellar :any` ensures that users who have installed Homebrew at a non-standard prefix will get the bottles.

## Handling `brew bump-formula-pr` PRs

### Formulae that exist in Homebrew/homebrew-core

When running on Linux, the `brew bump-formula-pr` command should raise pull
requests against the correct upstream macOS Homebrew-core repository. If a
pull request is raised against the Linuxbrew-core repository when an upstream
formula exists, please use the following message to direct users to the
correct repository:

> Thanks for your PR.
>
> However, this formula is not Linux-specific. Its new versions are merged from the [Homebrew/homebrew-core](https://github.com/Homebrew/homebrew-core) repository daily [as documented in CONTRIBUTING.md](https://github.com/Homebrew/linuxbrew-core/blob/HEAD/CONTRIBUTING.md). Please submit this change as a PR to that repository.
>
> We look forward to your PR against Homebrew/homebrew-core for the next version bump!

### Linux-only formulae

If the formula is a Linux-only formula, it either:
- will contain the line `depends_on :linux`
- won't have macOS bottles

If the user hasn't used `brew bump-formula-pr`, or is submitting
another change, you should request that they remove the `x86_64_linux`
bottle SHA line so that CI will build a bottle for the new version
correctly. If the bottle SHA isn't removed, CI will fail with the
following error:
> `--keep-old` was passed but there are changes in `sha256 => x86_64_linux`
