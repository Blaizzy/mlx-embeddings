#:  * `casks`
#:
#:  List all locally installable casks including short names.
#:

homebrew-casks() {
  local casks
  local sed_extended_regex_flag

  if [[ -n "$HOMEBREW_MACOS" ]]; then
    sed_extended_regex_flag="-E"
  else
    sed_extended_regex_flag="-r"
  fi

  casks="$( \
    find "$HOMEBREW_REPOSITORY/Library/Taps" \
         -maxdepth 4 -path '*/Casks/*.rb' | \
    sed "$sed_extended_regex_flag" \
      -e 's/\.rb//g' \
      -e 's_.*/Taps/(.*)/homebrew-_\1/_' \
      -e 's|/Casks/|/|' \
  )"
  local shortnames
  shortnames="$(echo "$casks" | cut -d "/" -f 3)"
  echo -e "$casks\n$shortnames" | \
    grep -v '^homebrew/cask' | \
    sort -uf
}
