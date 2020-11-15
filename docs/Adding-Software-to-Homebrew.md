# Adding Software to Homebrew

Are you missing your favorite software in Homebrew? Then you're the perfect person to resolve this problem.

Before you start, please check the open pull requests for [homebrew-core](https://github.com/Homebrew/homebrew-core/pulls) or [linuxbrew-core](https://github.com/Homebrew/linuxbrew-core/pulls), to make sure no one else beat you to the punch.

Next, you will want to go through the [Acceptable Formulae](Acceptable-Formulae.md) documentation to determine if the software is an appropriate addition to Homebrew. If you are creating a formula for an alternative version of software already in Homebrew (for example, a major/minor version that significantly differs from the existing version), be sure to read the [Versions](Versions.md) documentation to understand versioned formulae requirements.

If everything checks out, you're ready to get started on a new formula!

## Writing the formula
1. It's a good idea to find existing formulae in Homebrew that have similarities to the software you want to add. This will help you to understand how specific languages, build methods, etc. are typically handled.

1. If you're starting from scratch, the [`brew create` command](Manpage.md#create-options-url) can be used to produce a basic version of your formula. This command accepts a number of options and you may be able to save yourself some work by using an appropriate template option like `--python`.

1. You will now have to work to develop the boilerplate code from `brew create` into a fully-fledged formula. Your main references will be the [Formula Cookbook](Formula-Cookbook.md), similar formulae in Homebrew, and the upstream documentation for your chosen software. Be sure to also take note of the Homebrew documentation for writing [`Python`](Python-for-Formula-Authors.md) and [`Node`](Node-for-Formula-Authors.md) formulae, if applicable.  

1. Make sure you write a good test as part of your formula. Refer to the "[Add a test to the formula](Formula-Cookbook.md#add-a-test-to-the-formula)" section of the Cookbook for help with this.

1. Try to install the formula using `brew install --build-from-source <formula>`, where \<formula\> is the name of your formula. If any errors occur, correct your formula and attempt to install it again. The formula should install without errors by the end of this step.

If you're stuck, ask for help on GitHub or [Homebrew/discussions](https://github.com/homebrew/discussions/discussions). The maintainers are very happy to help but we also like to see that you've put effort into trying to find a solution first.

## Testing and auditing the formula

1. Run `brew audit --strict --new-formula --online <formula>` with your formula. If any errors occur, correct them in your formula and run the audit again. The audit should finish without any errors by the end of this step.

1. Run your formula's test using `brew test <formula>` with your formula. Your test should finish without any errors.

## Submitting the formula

You're finally ready to submit your formula to the [homebrew-core](https://github.com/Homebrew/homebrew-core/) or [linuxbrew-core](https://github.com/Homebrew/linuxbrew-core/) repository. If you haven't done this before, you can refer to the [How to Open a Pull Request](How-To-Open-a-Homebrew-Pull-Request.md) documentation for help. Maintainers will review the pull request and provide feedback about any areas that need to be addressed before the formula can be added to Homebrew.

If you've made it this far, congratulations on submitting a Homebrew formula! We appreciate the hard work you put into this and you can take satisfaction in knowing that your work may benefit other Homebrew users as well.
