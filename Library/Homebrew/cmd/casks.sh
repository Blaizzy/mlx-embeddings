#:  * `casks`
#:
#:  List all locally installable casks including short names.
#:

# HOMEBREW_LIBRARY is set in bin/brew
# shellcheck disable=SC2154
source "${HOMEBREW_LIBRARY}/Homebrew/items.sh"

homebrew-casks() {
  # HOMEBREW_CACHE is set by brew.sh
  # shellcheck disable=SC2154
  if [[ -z "${HOMEBREW_NO_INSTALL_FROM_API}" &&
        -f "${HOMEBREW_CACHE}/api/cask_names.txt" ]]
  then
    {
      cat "${HOMEBREW_CACHE}/api/cask_names.txt"
      echo
      homebrew-items '*/Casks/*\.rb' '.*/homebrew/homebrew-cask/.*' 's|/Casks/|/|' '^homebrew/cask'
    } | sort -uf
  else
    homebrew-items '*/Casks/*\.rb' '^\b$' 's|/Casks/|/|' '^homebrew/cask'
  fi
}
