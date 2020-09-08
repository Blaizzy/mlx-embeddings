# Migrating A Formula To A Tap

There are times when we may wish to migrate a formula from one tap into another tap. To do this:

1. Create a pull request to the new tap adding the formula file as-is from the original tap. Fix any test failures that may occur due to the stricter requirements for new formulae than existing formula (e.g. `brew audit --strict` must pass for that formula).
2. Create a pull request to the original tap deleting the formula file and add it to `tap_migrations.json` with a commit message like `gv: migrate to homebrew/core`.
3. Put a link for each pull request in the other pull request so the maintainers can merge them both at once.

Congratulations, you've moved a formula to a tap!

For Homebrew maintainers, formulae should only ever be migrated into and within the Homebrew organisation (e.g. from Homebrew/core to Homebrew/cask, or from a third-party tap to Homebrew/core), and never out of it.
