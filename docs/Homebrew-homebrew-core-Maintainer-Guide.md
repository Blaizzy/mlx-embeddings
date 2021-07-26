# Homebrew/homebrew-core Maintainer Guide

## Quick merge checklist

A detailed checklist can be found [below](#detailed-merge-checklist). This is all that really matters:

- Ensure the name seems reasonable.
- Add aliases.
- Ensure it uses `keg_only :provided_by_macos` if it already comes with macOS.
- Ensure it is not a library that can be installed with
  [gem](https://en.wikipedia.org/wiki/RubyGems),
  [cpan](https://en.wikipedia.org/wiki/Cpan) or
  [pip](https://pip.pypa.io/en/stable/).
- Ensure that any dependencies are accurate and minimal. We don't need to
  support every possible optional feature for the software.
- When bottles aren't required or affected, use the GitHub squash & merge workflow for a single-formula PR or rebase & merge workflow for a multiple-formulae PR. See the ["How to merge without bottles" section below](#how-to-merge-without-bottles)
for more details.
- Use `brew pr-publish` or `brew pr-pull` otherwise, which adds messages to auto-close pull requests and pull bottles built by the Brew Test Bot.
- Thank people for contributing.

Checking dependencies is important, because they will probably stick around
forever. Nobody really checks if they are necessary or not.

Depend on as little stuff as possible. Disable X11 functionality if possible.
For example, we build Wireshark, but not the heavy GUI.

Homebrew is about Unix software. Stuff that builds to an `.app` should
be in Homebrew Cask instead.

## Merging, rebasing, cherry-picking

For most PRs that make formula modifications, you can simply approve the PR and an automatic
merge (with bottles) will be performed by [@BrewTestBot](https://github.com/BrewTestBot).
See [Brew Test Bot For Core Contributors](Brew-Test-Bot-For-Core-Contributors.md) for more information.

Certain PRs may not be merged automatically by [@BrewTestBot](https://github.com/BrewTestBot),
even after they've been approved. This includes PRs with the `new formula`, `automerge-skip`,
and `linux-only` labels. To trigger a merge for these PRs, run `brew pr-publish`.

PRs modifying formulae that don't need bottles or making changes that don't
require new bottles to be pulled should use GitHub's squash & merge or rebase & merge workflows.
See the [table below](#how-to-merge-without-bottles) for more details.

Otherwise, you should use `brew pr-pull` (or `rebase`/`cherry-pick` contributions).

Don’t `rebase` until you finally `push`. Once `master` is pushed, you can’t
`rebase`: **you’re a maintainer now!**

Cherry-picking changes the date of the commit, which kind of sucks.

Don’t `merge` unclean branches. So if someone is still learning `git` and
their branch is filled with nonsensical merges, then `rebase` and squash
the commits. Our main branch history should be useful to other people,
not confusing.

Here’s a flowchart for managing a PR which is ready to merge:

![Flowchart for managing pull requests](assets/img/docs/managing-pull-requests.drawio.svg)

Only one maintainer is necessary to approve and merge the addition of a new or updated formula which passes CI. However, if the formula addition or update proves controversial the maintainer who adds it will be expected to answer requests and fix problems that arise with it in future.

### How to merge without bottles

Here are guidelines about when to use squash & merge versus rebase & merge. These options should only be used with PRs where bottles are not affected.

| | PR modifies a single formula | PR modifies multiple formulae |
|---|---|---|
| **Commits look good** | rebase & merge _or_ squash & merge | rebase & merge |
| **Commits need work** | squash & merge | manually merge using the command line |

## Naming

The name is the strictest item, because avoiding a later name change is
desirable.

Choose a name that’s the most common name for the project.
For example, we initially chose `objective-caml` but we should have chosen `ocaml`.
Choose what people say to each other when talking about the project.

Formulae that are also packaged by other package managers (e.g. Debian, Ubuntu) should be
named consistently (subject to minor differences due to Homebrew formula naming conventions).

Add other names as aliases as symlinks in `Aliases` in the tap root. Ensure the
name referenced on the homepage is one of these, as it may be different and have
underscores and hyphens and so on.

We now accept versioned formulae as long as they [meet the requirements](Versions.md).

## Testing

We need to at least check that it builds. Use the [Brew Test Bot](Brew-Test-Bot.md) for this.

Verify the formula works if possible. If you can’t tell (e.g. if it’s a
library) trust the original contributor, it worked for them, so chances are it
is fine. If you aren’t an expert in the tool in question, you can’t really
gauge if the formula installed the program correctly. At some point an expert
will come along, cry blue murder that it doesn’t work, and fix it. This is how
open source works. Ideally, request a `test do` block to test that
functionality is consistently available.

If the formula uses a repository, then the `url` parameter should have a
tag or revision. `url`s have versions and are stable (not yet
implemented!).

Don't merge any formula updates with failing `brew test`s. If a `test do` block
is failing it needs to be fixed. This may involve replacing more involved tests
with those that are more reliable. This is fine: false positives are better than
false negatives as we don't want to teach maintainers to merge red PRs. If the
test failure is believed to be due to a bug in `Homebrew/brew` or the CI system,
that bug must be fixed, or worked around in the formula to yield a passing test,
before the PR can be merged.

## Duplicates

We now accept stuff that comes with macOS as long as it uses `keg_only :provided_by_macos` to be keg-only by default.

## Removing formulae

Formulae that:

- work on at least 2/3 of our supported macOS versions in the default Homebrew prefix
- do not require patches rejected by upstream to work
- do not have known security vulnerabilities or CVEs for the version we package
- are shown to be still installed by users in our analytics with a `BuildError` rate of <25%

should not be removed from Homebrew. The exception to this rule are [versioned formulae](Versions.md) for which there are higher standards of usage and a maximum number of versions for a given formula.

For more information about deprecating, disabling and removing formulae, see the
[Deprecating, Disabling, and Removing Formulae page](Deprecating-Disabling-and-Removing-Formulae.md).

## Detailed merge checklist

The following checklist is intended to help maintainers decide on
whether to merge, request changes or close a PR. It also brings more
transparency for contributors in addition to the
[Acceptable Formulae](Acceptable-Formulae.md) requirements.

- previously opened active PRs, as we would like to be fair to contributors who came first
- patches/`inreplace` that have been applied to upstream and can be removed
- comments in formula around `url`, as we do skip some versions (for example [vim](https://github.com/Homebrew/homebrew-core/blob/359dbb190bb3776c4d6a1f603a271dd8f186f077/Formula/vim.rb#L4) or [v8](https://github.com/Homebrew/homebrew-core/blob/359dbb190bb3776c4d6a1f603a271dd8f186f077/Formula/v8.rb#L4))
- vendored resources that need updates (for example [emscripten](https://github.com/Homebrew/homebrew-core/commit/57126ac765c3ac5201ce53bcdebf7a0e19071eba))
- vendored dependencies (for example [certbot](https://github.com/Homebrew/homebrew-core/pull/42966/files))
- stable/announced release
  - some teams use odd minor release number for tests and even for stable releases
  - other teams drop new version with minor release 0 but promote it to stable only after a few minor releases
  - if the software uses only hosted version control (such as GitHub, GitLab or Bitbucket), the release should be tagged and if upstream marks latest/pre-releases, PR must use latest
- does changelog mention addition/removal of dependency and is it addressed in the PR
  - does formula depend on versioned formula (for example `python@3.7`, `go@1.10`, `erlang@17`) that can be upgraded
- commits
  - contain one formula change per commit
    - ask author to squash
    - rebase during merge
  - version update follows preferred message format for simple version updates: `foobar 7.3`
  - other fixes format is `foobar: fix flibble matrix`
- bottle block is not removed

  Suggested reply:
  ```
  Please keep bottle block in place, [@BrewTestBot](https://github.com/BrewTestBot) takes care of it.
  ```
- is there a test block for other than checking version or printing help? Consider asking to add one
- if CI failed
  - due to test block - paste relevant lines and add `test failure` label
  - due to build errors - paste relevant lines and add `build failure` label
  - due to other formulae needing revision bumps - suggest to use the following command:
    ```
    # in this example PR is for `libuv` formula and `urbit` needs revision bump
    brew bump-revision --message 'for libuv' urbit
    ```
    - make sure it is one commit per revision bump
- if CI is green and...
  - bottles need to be pulled, and...
    - the commits are correct, don't need changes, and BrewTestBot can merge it (doesn't have the label `automerge-skip`): approve the PR to trigger an automatic merge (use `brew pr-publish $PR_ID` to trigger manually in case of a new formula)
    - the commits are correct and don't need changes, but BrewTestBot can't merge it (has the label `automerge-skip`), use `brew pr-publish $PR_ID`
    - the commits need to be amended, use `brew pr-pull $PR_ID`, make changes, and `git push`
- don't forget to thank the contributor
  - celebrate the first-time contributors
- suggest to use `brew bump-formula-pr` next time if this was not the case
