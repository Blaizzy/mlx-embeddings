# Releases

Since Homebrew 1.0.0 most Homebrew users (those who haven't run a `dev-cmd` or
set `HOMEBREW_DEVELOPER=1`) require tags on the [Homebrew/brew repository](https://github.com/homebrew/brew)
in order to get new versions of Homebrew. There are a few steps in making a new
Homebrew release:

1. Check the [Homebrew/brew pull requests](https://github.com/homebrew/brew/pulls),
   [issues](https://github.com/homebrew/brew/issues),
   [Homebrew/core issues](https://github.com/homebrew/homebrew-core/issues) and
   [Homebrew/discussions (forum)](https://github.com/homebrew/discussions/discussions) to see if there is
   anything pressing that needs to be fixed or merged before the next release.
   If so, fix and merge these changes.
2. Ensure that no code changes have happened for at least a couple of hours (ideally 24 hours)
   and that you are confident there are no major regressions on the current `master`
   branch.
3. Run `brew release` to create a new draft release. For major or minor version bumps,
   pass `--major` or `--minor`, respectively.
4. Publish the draft release on [GitHub](https://github.com/Homebrew/brew/releases).

If this is a major or minor release (e.g. X.0.0 or X.Y.0) then there are a few more steps:

1. Before creating the tag you should delete any `odisabled` code, make any
   `odeprecated` code `odisabled` and add any new `odeprecations` that are
   desired.
2. Write up a release notes blog post to <https://brew.sh>
   e.g. [brew.sh#319](https://github.com/Homebrew/brew.sh/pull/319).
   This should use `brew release-notes` as input but have the wording adjusted
   to be more human readable and explain not just what has changed but why.
3. When the release has shipped and the blog post has been merged, tweet the
   blog post as the [@MacHomebrew Twitter account](https://twitter.com/MacHomebrew)
   or tweet it yourself and retweet it with the @MacHomebrew Twitter account
   (credentials are in 1Password).
4. Send the email to the Homebrew TinyLetter email list (credentials are in
   1Password).
5. Consider whether to submit it to other sources e.g. Hacker News, Reddit.

  - Pros: gets a wider reach and user feedback
  - Cons: negative comments are common and people take this as a chance to
          complain about Homebrew (regardless of their usage)
