#:  * `formulae`
#:
#:  List all locally installable formulae including short names.
#:

source "$HOMEBREW_LIBRARY/Homebrew/items.sh"

homebrew-formulae() {
  homebrew-items 'Casks' 's|/Formula/|/|' '^homebrew/core'
}
