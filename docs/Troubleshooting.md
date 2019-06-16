# Troubleshooting

**Run `brew update` twice and `brew doctor` (and fix all the warnings) *before* creating an issue!**

This document will help you check for common issues and make sure your issue has not already been reported.

## Check for common issues

* Read through the [Common Issues](Common-Issues.md).

## Check to see if the issue has been reported

* Search the [Homebrew/homebrew-core issue tracker](https://github.com/Homebrew/homebrew-core/issues) or [Homebrew/linuxbrew-core issue tracker](https://github.com/Homebrew/linuxbrew-core/issues) to see if someone else has already reported the same issue.
* If a formula that has failed to build is part of a non-core tap or a cask is part of [homebrew/cask](https://github.com/Homebrew/homebrew-cask/issues) check those issue trackers instead.

## Create an issue

If your problem hasn't been solved or reported, then create an issue:

1. Upload debugging information to a [Gist](https://gist.github.com):

  * If you had a formula-related problem: run `brew gist-logs <formula>` (where `<formula>` is the name of the formula).
  * If you encountered a non-formula problem: upload the output of `brew config` and `brew doctor` to a new [Gist](https://gist.github.com).

2. Create a new issue on the [Homebrew/homebrew-core issue tracker](https://github.com/Homebrew/homebrew-core/issues/new/choose) or [Homebrew/linuxbrew-core issue tracker](https://github.com/Homebrew/linuxbrew-core/issues/new/choose)

  * Give your issue a descriptive title which includes the formula name (if applicable) and the version of macOS or Linux you are using. For example, if a formula fails to build, title your issue "\<formula> failed to build on \<10.x>", where "\<formula>" is the name of the formula that failed to build, and "\<10.x>" is the version of macOS or Linux you are using.
  * Include the URL output by `brew gist-logs <formula>` (if applicable).
  * Include links to any additional Gists you may have created (such as for the output of `brew config` and `brew doctor`).
