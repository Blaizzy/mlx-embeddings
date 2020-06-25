# Brew Test Bot For Core Contributors

If a build has run and passed on `brew test-bot` then it can be used to quickly bottle formulae.

## Bottling

If a pull request is correct and doesn't need any modifications to commit messages or otherwise:

1. Ensure the job has already completed successfully.
2. Run `brew pr-publish 12345` where `12345` is the pull request number (or URL).
    - Approving a PR for an existing formula will automatically publish the bottles and close the PR, taking care of this step.
3. Watch the [actions queue](https://github.com/Homebrew/homebrew-core/actions) to ensure your job finishes. BrewTestBot will usually notify you of failures with a ping as well.

If a pull request needs changes to the commit messages:

1. Ensure the job has already completed successfully.
2. Run `brew pr-pull 12345` where `12345` is the pull request number (or URL).
3. Amend any relevant commits, then run `git push` to push the commits.
