#:  * `setup-ruby [command]`
#:
#:  Installs and configures Homebrew's Ruby.
#:  If `command` is passed, it will only run Bundler if necessary for that
#:  command.
#:

# HOMEBREW_LIBRARY is from the user environment.
# HOMEBREW_RUBY_PATH is set by utils/ruby.sh
# RUBY_DISABLE_OPTIONS is set by brew.sh
# HOMEBREW_BREW_FILE is set by extend/ENV/super.rb
# shellcheck disable=SC2154
homebrew-setup-ruby() {
  source "${HOMEBREW_LIBRARY}/Homebrew/utils/ruby.sh"
  setup-ruby-path

  if [[ -z "${HOMEBREW_DEVELOPER}" ]]
  then
    return
  fi

  # Avoid running Bundler if the command doesn't need it.
  local command="$1"
  if [[ -n "${command}" ]]
  then
    source "${HOMEBREW_LIBRARY}/Homebrew/command_path.sh"

    command_path="$(homebrew-command-path "${command}")"
    if [[ -n "${command_path}" && "${command_path}" != *"/dev-cmd/"* ]]
    then
      return
    fi

    if ! grep -q "Homebrew.install_bundler_gems\!" "${command_path}"
    then
      return
    fi
  fi

  GEM_VERSION="$("${HOMEBREW_RUBY_PATH}" "${HOMEBREW_RUBY_DISABLE_OPTIONS}" /dev/stdin <<<'require "rbconfig"; puts RbConfig::CONFIG["ruby_version"]')"
  echo "${GEM_VERSION}"
  GEM_HOME="${HOMEBREW_LIBRARY}/Homebrew/vendor/bundle/ruby/${GEM_VERSION}"
  BUNDLE_GEMFILE="${HOMEBREW_LIBRARY}/Homebrew/Gemfile"

  export GEM_HOME
  export BUNDLE_GEMFILE

  if ! bundle check &>/dev/null
  then
    "${HOMEBREW_BREW_FILE}" install-bundler-gems
  fi
}
