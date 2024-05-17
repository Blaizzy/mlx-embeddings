# does the quickest output of brew list possible for no named arguments.
# HOMEBREW_CELLAR, HOMEBREW_PREFIX are set by brew.sh
# shellcheck disable=SC2154
homebrew-list() {
  case "$1" in
    # check we actually have list and not e.g. listsomething
    list) ;;
    list*) return 1 ;;
    *) ;;
  esac

  local ls_args=()
  local formula=""
  local cask=""

  for arg in "$@"
  do
    case "${arg}" in
      # check for flags passed to ls
      -1 | -l | -r | -t) ls_args+=("${arg}") ;;
      --formula | --formulae) formula=1 ;;
      --cask | --casks) cask=1 ;;
      # reject all other flags
      -* | *) return 1 ;;
    esac
  done

  local tty
  if [[ -t 1 ]]
  then
    tty=1
  fi

  local error_string="LS_ERRORED"
  if [[ -z "${cask}" && -d "${HOMEBREW_CELLAR}" ]]
  then
    if [[ -n "${tty}" && -z "${formula}" ]]
    then
      ohai "Formulae"
    fi

    local formula_output
    formula_output="$(ls "${ls_args[@]}" "${HOMEBREW_CELLAR}" || echo "${error_string}")"
    if [[ "${formula_output}" == "${error_string}" ]]
    then
      exit 1
    elif [[ -n "${formula_output}" ]]
    then
      echo "${formula_output}"
    fi

    if [[ -n "${tty}" && -z "${formula}" ]]
    then
      echo
    fi
  fi

  if [[ -z "${formula}" && -d "${HOMEBREW_CASKROOM}" ]]
  then
    if [[ -n "${tty}" && -z "${cask}" ]]
    then
      ohai "Casks"
    fi

    local cask_output
    cask_output="$(ls "${ls_args[@]}" "${HOMEBREW_CASKROOM}" || echo "${error_string}")"
    if [[ "${cask_output}" == "${error_string}" ]]
    then
      exit 1
    elif [[ -n "${cask_output}" ]]
    then
      echo "${cask_output}"
    fi

    return 0
  fi
}
