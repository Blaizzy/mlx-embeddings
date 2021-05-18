#:  * `formulae`
#:
#:  List all locally installable formulae including short names.
#:

# HOMEBREW_LIBRARY is set by bin/brew
# shellcheck disable=SC2154
source "${HOMEBREW_LIBRARY}/Homebrew/items.sh"

homebrew-formulae() {
  homebrew-items 'Casks' 's|/Formula/|/|' '^homebrew/core'
}
