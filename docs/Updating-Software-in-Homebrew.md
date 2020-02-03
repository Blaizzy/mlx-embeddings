# Updating Software in Homebrew

Did you find something in Homebrew that wasn't the latest version? You can help yourself and others by submitting a pull request to update the formula.

First, check the pull requests in the [homebrew-core](https://github.com/Homebrew/homebrew-core/pulls) or [linuxbrew-core](https://github.com/Homebrew/linuxbrew-core/pulls) repositories to make sure there isn't already an open PR. You may also want to look through closed pull requests for the formula, as sometimes formulae run into problems preventing them from being updated and it's better to be aware of any issues before putting significant effort into an update.

The guide on [opening a pull request](How-To-Open-a-Homebrew-Pull-Request.md#submit-a-new-version-of-an-existing-formula) should really be all you need, this will explain how to easily change the url to point to the latest version and that's really all you need. If you want to read up on `bump-formula-pr` before using it you could check [the manpage](Manpage.md#bump-formula-pr-options-formula).

You can look back at previous pull requests that updated the formula to see how others have handled things in the past but be sure to look at a variety of PRs. Sometimes formulae are not updated properly (for example, running `bump-formula-pr` on a Python formula that needs dependency updates), so you may need to use your judgment to determine how to proceed.

Once you've created the pull request in the appropriate Homebrew repository your commit(s) will be tested on our continuous integration servers, showing a green check mark when everything passed or a red X when there were failures. Maintainers will review your pull request and provide feedback about any changes that need to be made before it can be merged.

We appreciate your help in keeping Homebrew's formulae up to date as new versions of software are released!
