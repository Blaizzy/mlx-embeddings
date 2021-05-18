#:  @hide_from_man_page
#:  * `vendor-install` [<target>]
#:
#:  Install Homebrew's portable Ruby.

# HOMEBREW_CURLRC, HOMEBREW_LIBRARY, HOMEBREW_STDERR is from the user environment
# HOMEBREW_CACHE, HOMEBREW_CURL, HOMEBREW_LINUX, HOMEBREW_LINUX_MINIMUM_GLIBC_VERSION, HOMEBREW_MACOS,
# HOMEBREW_MACOS_VERSION_NUMERIC and HOMEBREW_PROCESSOR are set by brew.sh
# shellcheck disable=SC2154
source "${HOMEBREW_LIBRARY}/Homebrew/utils/lock.sh"

VENDOR_DIR="${HOMEBREW_LIBRARY}/Homebrew/vendor"

# Built from https://github.com/Homebrew/homebrew-portable-ruby.
if [[ -n "${HOMEBREW_MACOS}" ]]
then
  if [[ "${HOMEBREW_PROCESSOR}" = "Intel" ]]
  then
    ruby_FILENAME="portable-ruby-2.6.3_2.yosemite.bottle.tar.gz"
    ruby_SHA="b065e5e3783954f3e65d8d3a6377ca51649bfcfa21b356b0dd70490f74c6bd86"
  fi
elif [[ -n "${HOMEBREW_LINUX}" ]]
then
  case "${HOMEBREW_PROCESSOR}" in
    x86_64)
      ruby_FILENAME="portable-ruby-2.6.3_2.x86_64_linux.bottle.tar.gz"
      ruby_SHA="97e639a64dcec285392b53ad804b5334c324f1d2a8bdc2b5087b8bf8051e332f"
      ;;
    *)
  esac
fi

# Dynamic variables can't be detected by shellcheck
# shellcheck disable=SC2034
if [[ -n "${ruby_SHA}" && -n "${ruby_FILENAME}" ]]
then
  ruby_URL="https://ghcr.io/v2/homebrew/portable-ruby/portable-ruby/blobs/sha256:${ruby_SHA}"
  ruby_URL2="https://github.com/Homebrew/homebrew-portable-ruby/releases/download/2.6.3_2/${ruby_FILENAME}"
fi

check_linux_glibc_version() {
  if [[ -z ${HOMEBREW_LINUX} || -z ${HOMEBREW_LINUX_MINIMUM_GLIBC_VERSION} ]]
  then
    return 0
  fi

  local glibc_version
  local glibc_version_major
  local glibc_version_minor

  local minimum_required_major=${HOMEBREW_LINUX_MINIMUM_GLIBC_VERSION%.*}
  local minimum_required_minor=${HOMEBREW_LINUX_MINIMUM_GLIBC_VERSION#*.}

  if [[ $(/usr/bin/ldd --version) =~ \ [0-9]\.[0-9]+ ]]
  then
    glibc_version=${BASH_REMATCH[0]// /}
    glibc_version_major=${glibc_version%.*}
    glibc_version_minor=${glibc_version#*.}
    if (( glibc_version_major < minimum_required_major || glibc_version_minor < minimum_required_minor ))
    then
      odie "Vendored tools require system Glibc ${HOMEBREW_LINUX_MINIMUM_GLIBC_VERSION} or later (yours is ${glibc_version})."
    fi
  else
    odie "Failed to detect system Glibc version."
  fi
}

# Execute the specified command, and suppress stderr unless HOMEBREW_STDERR is set.
quiet_stderr() {
  if [[ -z "${HOMEBREW_STDERR}" ]]; then
    command "$@" 2>/dev/null
  else
    command "$@"
  fi
}

fetch() {
  local -a curl_args
  local sha
  local temporary_path

  curl_args=()

  # do not load .curlrc unless requested (must be the first argument)
  # HOMEBREW_CURLRC isn't misspelt here
  # shellcheck disable=SC2153
  if [[ -z "${HOMEBREW_CURLRC}" ]]
  then
    curl_args[${#curl_args[*]}]="-q"
  fi

  # Authorization is needed for GitHub Packages but harmless on GitHub Releases
  curl_args+=(
    --fail
    --remote-time
    --location
    --user-agent "${HOMEBREW_USER_AGENT_CURL}"
    --header "Authorization: Bearer QQ=="
  )

  if [[ -n "${HOMEBREW_QUIET}" ]]
  then
    curl_args[${#curl_args[*]}]="--silent"
  elif [[ -z "${HOMEBREW_VERBOSE}" ]]
  then
    curl_args[${#curl_args[*]}]="--progress-bar"
  fi

  if [[ "${HOMEBREW_MACOS_VERSION_NUMERIC}" -lt "100600" ]]
  then
    curl_args[${#curl_args[*]}]="--insecure"
  fi

  temporary_path="${CACHED_LOCATION}.incomplete"

  mkdir -p "${HOMEBREW_CACHE}"
  [[ -n "${HOMEBREW_QUIET}" ]] || ohai "Downloading ${VENDOR_URL}" >&2
  if [[ -f "${CACHED_LOCATION}" ]]
  then
    [[ -n "${HOMEBREW_QUIET}" ]] || echo "Already downloaded: ${CACHED_LOCATION}" >&2
  else
    if [[ -f "${temporary_path}" ]]
    then
      # HOMEBREW_CURL is set by brew.sh (and isn't mispelt here)
      # shellcheck disable=SC2153
      "${HOMEBREW_CURL}" "${curl_args[@]}" -C - "${VENDOR_URL}" -o "${temporary_path}"
      if [[ $? -eq 33 ]]
      then
        [[ -n "${HOMEBREW_QUIET}" ]] || echo "Trying a full download" >&2
        rm -f "${temporary_path}"
        "${HOMEBREW_CURL}" "${curl_args[@]}" "${VENDOR_URL}" -o "${temporary_path}"
      fi
    else
      "${HOMEBREW_CURL}" "${curl_args[@]}" "${VENDOR_URL}" -o "${temporary_path}"
    fi

    if [[ ! -f "${temporary_path}" ]]
    then
      [[ -n "${HOMEBREW_QUIET}" ]] || ohai "Downloading ${VENDOR_URL2}" >&2
      "${HOMEBREW_CURL}" "${curl_args[@]}" "${VENDOR_URL2}" -o "${temporary_path}"
    fi

    if [[ ! -f "${temporary_path}" ]]
    then
      odie <<EOS
Failed to download ${VENDOR_URL} and ${VENDOR_URL2}!

Do not file an issue on GitHub about this; you will need to figure out for
yourself what issue with your internet connection restricts your access to
GitHub (used for Homebrew updates and binary packages).
EOS
    fi

    trap '' SIGINT
    mv "${temporary_path}" "${CACHED_LOCATION}"
    trap - SIGINT
  fi

  if [[ -x "/usr/bin/shasum" ]]
  then
    sha="$(/usr/bin/shasum -a 256 "${CACHED_LOCATION}" | cut -d' ' -f1)"
  elif [[ -x "$(type -P sha256sum)" ]]
  then
    sha="$(sha256sum "${CACHED_LOCATION}" | cut -d' ' -f1)"
  elif [[ -x "$(type -P ruby)" ]]
  then
    sha="$(ruby <<EOSCRIPT
            require 'digest/sha2'
            digest = Digest::SHA256.new
            File.open('${CACHED_LOCATION}', 'rb') { |f| digest.update(f.read) }
            puts digest.hexdigest
EOSCRIPT
)"
  else
    odie "Cannot verify checksum ('shasum' or 'sha256sum' not found)!"
  fi

  if [[ "${sha}" != "${VENDOR_SHA}" ]]
  then
    odie <<EOS
Checksum mismatch.
Expected: ${VENDOR_SHA}
  Actual: ${sha}
 Archive: ${CACHED_LOCATION}
To retry an incomplete download, remove the file above.
EOS
  fi
}

install() {
  local tar_args

  if [[ -n "${HOMEBREW_VERBOSE}" ]]
  then
    tar_args="xvzf"
  else
    tar_args="xzf"
  fi

  mkdir -p "${VENDOR_DIR}/portable-${VENDOR_NAME}"
  safe_cd "${VENDOR_DIR}/portable-${VENDOR_NAME}"

  trap '' SIGINT

  if [[ -d "${VENDOR_VERSION}" ]]
  then
    mv "${VENDOR_VERSION}" "${VENDOR_VERSION}.reinstall"
  fi

  safe_cd "${VENDOR_DIR}"
  [[ -n "${HOMEBREW_QUIET}" ]] || ohai "Pouring ${VENDOR_FILENAME}" >&2
  tar "${tar_args}" "${CACHED_LOCATION}"
  safe_cd "${VENDOR_DIR}/portable-${VENDOR_NAME}"

  if quiet_stderr "./${VENDOR_VERSION}/bin/${VENDOR_NAME}" --version >/dev/null
  then
    ln -sfn "${VENDOR_VERSION}" current
    if [[ -d "${VENDOR_VERSION}.reinstall" ]]
    then
      rm -rf "${VENDOR_VERSION}.reinstall"
    fi
  else
    rm -rf "${VENDOR_VERSION}"
    if [[ -d "${VENDOR_VERSION}.reinstall" ]]
    then
      mv "${VENDOR_VERSION}.reinstall" "${VENDOR_VERSION}"
    fi
    odie "Failed to install ${VENDOR_NAME} ${VENDOR_VERSION}!"
  fi

  trap - SIGINT
}

homebrew-vendor-install() {
  local option
  local url_var
  local sha_var

  for option in "$@"
  do
    case "${option}" in
      -\?|-h|--help|--usage) brew help vendor-install; exit $? ;;
      --verbose)             HOMEBREW_VERBOSE=1 ;;
      --quiet)               HOMEBREW_QUIET=1 ;;
      --debug)               HOMEBREW_DEBUG=1 ;;
      --*)                   ;;
      -*)
        [[ "${option}" = *v* ]] && HOMEBREW_VERBOSE=1
        [[ "${option}" = *q* ]] && HOMEBREW_QUIET=1
        [[ "${option}" = *d* ]] && HOMEBREW_DEBUG=1
        ;;
      *)
        [[ -n "${VENDOR_NAME}" ]] && odie "This command does not take multiple vendor targets!"
        VENDOR_NAME="${option}"
        ;;
    esac
  done

  [[ -z "${VENDOR_NAME}" ]] && odie "This command requires a vendor target!"
  [[ -n "${HOMEBREW_DEBUG}" ]] && set -x
  check_linux_glibc_version

  filename_var="${VENDOR_NAME}_FILENAME"
  sha_var="${VENDOR_NAME}_SHA"
  url_var="${VENDOR_NAME}_URL"
  url2_var="${VENDOR_NAME}_URL2"
  VENDOR_FILENAME="${!filename_var}"
  VENDOR_SHA="${!sha_var}"
  VENDOR_URL="${!url_var}"
  VENDOR_URL2="${!url2_var}"
  VENDOR_VERSION="$(<"${VENDOR_DIR}/portable-${VENDOR_NAME}-version")"

  if [[ -z "${VENDOR_URL}" || -z "${VENDOR_SHA}" ]]
  then
    odie "No Homebrew ${VENDOR_NAME} ${VENDOR_VERSION} available for ${HOMEBREW_PROCESSOR} processors!"
  fi

  CACHED_LOCATION="${HOMEBREW_CACHE}/${VENDOR_FILENAME}"

  lock "vendor-install-${VENDOR_NAME}"
  fetch
  install
}
