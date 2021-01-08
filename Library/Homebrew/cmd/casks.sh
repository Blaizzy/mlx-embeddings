#:  * `casks`
#:
#:  List all locally installable casks including short names.
#:

# Don't need shellcheck to follow the `source`.
# shellcheck disable=SC1090
source "$HOMEBREW_LIBRARY/Homebrew/items.sh"

homebrew-casks() {
  homebrew-items 'Formula' 's|/Casks/|/|' '^homebrew/cask'
}
