#:  * `formulae`
#:
#:  List all locally installable formulae including short names.
#:

# HOMEBREW_LIBRARY is set by bin/brew
# shellcheck disable=SC2154
source "${HOMEBREW_LIBRARY}/Homebrew/items.sh"

homebrew-formulae() {
  # HOMEBREW_CACHE is set by brew.sh
  # shellcheck disable=SC2154
  if [[ -z "${HOMEBREW_NO_INSTALL_FROM_API}" &&
        -f "${HOMEBREW_CACHE}/api/formula_names.txt" ]]
  then
    {
      cat "${HOMEBREW_CACHE}/api/formula_names.txt"
      echo
      homebrew-items '*\.rb' '.*Casks(/.*|$)|.*/homebrew/homebrew-core/.*' 's|/Formula/|/|' '^homebrew/core'
    } | sort -uf
  else
    homebrew-items '*\.rb' '.*Casks(/.*|$)' 's|/Formula/|/|' '^homebrew/core'
  fi
}
