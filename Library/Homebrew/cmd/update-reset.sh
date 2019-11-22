#:  * `update-reset` [<repository>]
#:
#:  Fetch and reset Homebrew and all tap repositories (or any specified <repository>) using `git`(1) to their latest `origin/master`.
#:
#:  *Note:* this will destroy all your uncommitted or committed changes.

homebrew-update-reset() {
  local DIR
  local -a REPOS=()

  for option in "$@"
  do
    case "$option" in
      -\?|-h|--help|--usage)          brew help update-reset; exit $? ;;
      --debug)                        HOMEBREW_DEBUG=1 ;;
      -*)
        [[ "$option" = *d* ]] && HOMEBREW_DEBUG=1
        ;;
      *)
        REPOS+=("$option")
        ;;
    esac
  done

  if [[ -n "$HOMEBREW_DEBUG" ]]
  then
    set -x
  fi

  if [[ -z "${REPOS[*]}" ]]
  then
    REPOS+=("$HOMEBREW_REPOSITORY" "$HOMEBREW_LIBRARY"/Taps/*/*)
  fi

  for DIR in "${REPOS[@]}"
  do
    [[ -d "$DIR/.git" ]] || continue
    cd "$DIR" || continue
    ohai "Fetching $DIR..."
    git fetch --force --tags origin
    echo

    ohai "Resetting $DIR..."
    git checkout --force -B master origin/master
    echo
  done
}
