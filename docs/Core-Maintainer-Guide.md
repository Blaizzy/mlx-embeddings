# Homebrew/homebrew-core Merge Checklist

The following checklist is intended to help maintainers decide on
whether to merge, request changes or close a PR. It also brings more
transparency for contributors in addition to
[Acceptable Formulae](Acceptable-Formulae.md) requirements.

This is a guiding principle. As a maintainer, you can make a call to either
request changes from a contributor or help them out based on their comfort and
previous contributions. Remember, as a team we
[Prioritise Maintainers Over Users](Maintainers-Avoiding-Burnout.md) to avoid
burnout.

This is a more practical checklist; it should be used after you get familiar with
[Maintainer Guidelines](Maintainer-Guidelines.md).

## Checklist

Check for:

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
  - formula `bottle :unneeded`, you can merge it through GitHub UI
  - bottles need to be pulled, and...
    - the commits are correct, don't need changes, and BrewTestBot can merge it (doesn't have the label `automerge-skip`): approve the PR to trigger an automatic merge (use `brew pr-publish $PR_ID` to trigger manually in case of a new formula)
    - the commits are correct and don't need changes, but BrewTestBot can't merge it (has the label `automerge-skip`), use `brew pr-publish $PR_ID`
    - the commits need to be amended, use `brew pr-pull $PR_ID`, make changes, and `git push`
- don't forget to thank the contributor
  - celebrate the first-time contributors
- suggest to use `brew bump-formula-pr` next time if this was not the case
