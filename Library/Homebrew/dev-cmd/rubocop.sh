#:  * `rubocop`
#:
#:  Installs, configures and runs Homebrew's `rubocop`.

# HOMEBREW_LIBRARY is from the user environment.
# HOMEBREW_RUBY_PATH is set by utils/ruby.sh
# RUBY_DISABLE_OPTIONS is set by brew.sh
# HOMEBREW_BREW_FILE is set by extend/ENV/super.rb
# shellcheck disable=SC2154
homebrew-rubocop() {
  source "${HOMEBREW_LIBRARY}/Homebrew/utils/ruby.sh"
  setup-ruby-path

  GEM_VERSION="$("${HOMEBREW_RUBY_PATH}" "${RUBY_DISABLE_OPTIONS}" -rrbconfig -e 'puts RbConfig::CONFIG["ruby_version"]')"
  GEM_HOME="${HOMEBREW_LIBRARY}/Homebrew/vendor/bundle/ruby/${GEM_VERSION}"

  if ! [[ -f "${GEM_HOME}/bin/rubocop" ]]; then
    "${HOMEBREW_BREW_FILE}" install-bundler-gems
  fi

  export GEM_HOME
  export PATH="${GEM_HOME}/bin:${PATH}"

  RUBOCOP="${HOMEBREW_LIBRARY}/Homebrew/utils/rubocop.rb"
  exec "${HOMEBREW_RUBY_PATH}" "${RUBOCOP}" "$@"
}
