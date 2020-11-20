#:  * `formulae`
#:
#:  List all locally installable formulae including short names.
#:

homebrew-formulae() {
  local formulae
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
    sed -E -e 's/\.rb//g' \
           -e 's_.*/Taps/(.*)/(home|linux)brew-_\1/_' \
           -e 's|/Formula/|/|' \
  )"
  local shortnames
  shortnames="$(echo "$formulae" | cut -d "/" -f 3)"
  echo -e "$formulae \n $shortnames" | \
    grep -v '^homebrew/core' | \
    sort -uf
}
