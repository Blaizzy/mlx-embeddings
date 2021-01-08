#:  * `casks`
#:
#:  List all locally installable casks including short names.
#:

source "$HOMEBREW_LIBRARY/Homebrew/items.sh"

homebrew-casks() {
  homebrew-items 'Formula' 's|/Casks/|/|' '^homebrew/cask'
}
