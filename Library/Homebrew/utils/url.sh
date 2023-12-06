# url_get <URL> [key] [key] ...
# where:
#   URL ~ [scheme://][user[:pass]@]host[:port][path]
#   key is one of the element composing the URL
url_get() {
  [[ ${#} -gt 0 ]] || return

  local url="${1}"
  [[ -z "${url}" ]] && return
  shift

  local _uphpp="${url#*://}"
  local _scheme="${url%%://*}"
  if [[ "${url}" == "${_uphpp}" ]]; then
    _scheme=""
  fi

  local _hpp="${_uphpp#*@}"
  local _up="${_uphpp%%@*}"
  if [[ "${_uphpp}" == "${_hpp}" ]]; then
    local _user=""
    local _pass=""
  else
    local _user="${_up%:*}"
    local _pass="${_up#*:}"
    if [[ "${_user}" == "${_up}" ]]; then
      _pass=""
    fi
  fi

  local _hp="${_hpp%%/*}"
  # TODO: split path from arguments (path?arg=...&....)
  local _path="${_hpp#*/}"
  if [[ "${_hp}" == "${_hpp}" ]]; then
    _path=""
  elif [[ -z "${_path}" ]]; then
    _path="/"
  fi

  local _host="${_hp%:*}"
  local _port="${_hp#*:}"
  if [[ "${_hp}" == "${_host}" ]]; then
    _port=""
  fi

  while [[ ${#} -gt 0 ]]; do
    echo -n \""$(eval echo -n "\${_${1}}")"\"
    [[ ${#} -gt 1 ]] && echo -n "${IFS:- }"
    shift
  done
}

# EoF
