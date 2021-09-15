homebrew-items() {
  local items
  local sed_extended_regex_flag
  local find_filter="$1"
  local sed_filter="$2"
  local grep_filter="$3"

  # HOMEBREW_MACOS is set by brew.sh
  # shellcheck disable=SC2154
  if [[ -n "${HOMEBREW_MACOS}" ]]
  then
    sed_extended_regex_flag="-E"
  else
    sed_extended_regex_flag="-r"
  fi

  # HOMEBREW_REPOSITORY is set by brew.sh
  # shellcheck disable=SC2154
  items="$(
    find "${HOMEBREW_REPOSITORY}/Library/Taps" \
      -type d \( \
      -name "${find_filter}" -o \
      -name cmd -o \
      -name .github -o \
      -name lib -o \
      -name spec -o \
      -name vendor \
      \) \
      -prune -false -o -name '*\.rb' |
      sed "${sed_extended_regex_flag}" \
        -e 's/\.rb//g' \
        -e 's_.*/Taps/(.*)/(home|linux)brew-_\1/_' \
        -e "${sed_filter}"
  )"
  local shortnames
  shortnames="$(echo "${items}" | cut -d "/" -f 3)"
  echo -e "${items}\n${shortnames}" |
    grep -v "${grep_filter}" |
    sort -uf
}
