#:  * `formulae`
#:
#:  List all locally installable formulae including short names.
#:

# Don't need shellcheck to follow the `source`.
# shellcheck disable=SC1090
source "$HOMEBREW_LIBRARY/Homebrew/items.sh"

homebrew-formulae() {
  homebrew-items 'Casks' 's|/Formula/|/|' '^homebrew/core'
}
