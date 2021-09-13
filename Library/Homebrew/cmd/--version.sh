#:  * `--version`, `-v`
#:
#:  Print the version numbers of Homebrew, Homebrew/homebrew-core and Homebrew/homebrew-cask (if tapped) to standard output.

# HOMEBREW_CORE_REPOSITORY, HOMEBREW_CASK_REPOSITORY, HOMEBREW_VERSION are set by brew.sh
# shellcheck disable=SC2154
version_string() {
  local repo="$1"
  if ! [[ -d "${repo}" ]]
  then
    echo "N/A"
    return
  fi

  local pretty_revision
  pretty_revision="$(git -C "${repo}" rev-parse --short --verify --quiet HEAD)"
  if [[ -z "${pretty_revision}" ]]
  then
    echo "(no Git repository)"
    return
  fi

  local git_last_commit_date
  git_last_commit_date="$(git -C "${repo}" show -s --format='%cd' --date=short HEAD)"
  echo "(git revision ${pretty_revision}; last commit ${git_last_commit_date})"
}

homebrew-version() {
  echo "Homebrew ${HOMEBREW_VERSION}"
  echo "Homebrew/homebrew-core $(version_string "${HOMEBREW_CORE_REPOSITORY}")"

  if [[ -d "${HOMEBREW_CASK_REPOSITORY}" ]]
  then
    echo "Homebrew/homebrew-cask $(version_string "${HOMEBREW_CASK_REPOSITORY}")"
  fi
}
