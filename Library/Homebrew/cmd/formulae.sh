#:  * `formulae`
#:
#:  List all locally installable formulae including short names.
#:

# HOMEBREW_LIBRARY is set by bin/brew
# shellcheck disable=SC2154
source "${HOMEBREW_LIBRARY}/Homebrew/items.sh"

homebrew-formulae() {
  local formulae
  formulae="$(homebrew-items '*\.rb' 'Casks' 's|/Formula/|/|' '^homebrew/core')"

  # HOMEBREW_CACHE is set by brew.sh
  # shellcheck disable=SC2154
  if [[ -z "${HOMEBREW_NO_INSTALL_FROM_API}" &&
        -f "${HOMEBREW_CACHE}/api/formula.json" ]]
  then
    local api_formulae
    api_formulae="$(ruby -e "require 'json'; JSON.parse(File.read('${HOMEBREW_CACHE}/api/formula.json')).each { |f| puts f['name'] }" 2>/dev/null)"
    formulae="$(echo -e "${formulae}\n${api_formulae}" | sort -uf | grep .)"
  fi

  echo "${formulae}"
}
