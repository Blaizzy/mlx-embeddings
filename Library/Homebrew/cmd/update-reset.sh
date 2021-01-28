#:  * `update-reset` [<repository> ...]
#:
#:  Fetch and reset Homebrew and all tap repositories (or any specified <repository>) using `git`(1) to their latest `origin/HEAD`.
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
    ohai "Fetching $DIR..."
    git -C "$DIR" fetch --force --tags origin
    git -C "$DIR" remote set-head origin --auto >/dev/null
    echo

    ohai "Resetting $DIR..."
    head="$(git -C "$DIR" symbolic-ref refs/remotes/origin/HEAD)"
    head="${head#refs/remotes/origin/}"
    git -C "$DIR" checkout --force -B "$head" origin/HEAD
    echo
  done
}
