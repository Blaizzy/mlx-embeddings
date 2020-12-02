#:  * `formulae`
#:
#:  List all locally installable formulae including short names.
#:

homebrew-formulae() {
  local formulae
  local sed_extended_regex_flag

  if [[ -n "$HOMEBREW_MACOS" ]]; then
    sed_extended_regex_flag="-E"
  else
    sed_extended_regex_flag="-r"
  fi

  formulae="$( \
    find "$HOMEBREW_REPOSITORY/Library/Taps" \
         -type d \( \
           -name Casks -o \
           -name cmd -o \
           -name .github -o \
           -name lib -o \
           -name spec -o \
           -name vendor \
          \) \
         -prune -false -o -name '*\.rb' | \
    sed "$sed_extended_regex_flag" \
      -e 's/\.rb//g' \
      -e 's_.*/Taps/(.*)/(home|linux)brew-_\1/_' \
      -e 's|/Formula/|/|' \
  )"
  local shortnames
  shortnames="$(echo "$formulae" | cut -d "/" -f 3)"
  echo -e "$formulae\n$shortnames" | \
    grep -v '^homebrew/core' | \
    sort -uf
}
