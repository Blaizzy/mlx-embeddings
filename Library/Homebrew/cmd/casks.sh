#:  * `casks`
#:
#:  List all locally installable casks including short names.
#:

# HOMEBREW_LIBRARY is set in bin/brew
# shellcheck disable=SC2154
source "${HOMEBREW_LIBRARY}/Homebrew/items.sh"

homebrew-casks() {
  homebrew-items 'Formula' 's|/Casks/|/|' '^homebrew/cask'
}
