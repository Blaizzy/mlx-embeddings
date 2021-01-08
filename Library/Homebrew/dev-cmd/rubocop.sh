#:  * `rubocop`
#:
#:  Installs, configures and runs Homebrew's `rubocop`.

homebrew-rubocop() {
  # Don't need shellcheck to follow this `source`.
  # shellcheck disable=SC1090
  source "$HOMEBREW_LIBRARY/Homebrew/utils/ruby.sh"
  setup-ruby-path

  GEM_VERSION="$("$HOMEBREW_RUBY_PATH" "$RUBY_DISABLE_OPTIONS" -rrbconfig -e 'puts RbConfig::CONFIG["ruby_version"]')"
  GEM_HOME="$HOMEBREW_LIBRARY/Homebrew/vendor/bundle/ruby/$GEM_VERSION"

  if ! [[ -f "$GEM_HOME/bin/rubocop" ]]; then
    "$HOMEBREW_BREW_FILE" install-bundler-gems
  fi

  export GEM_HOME
  export PATH="$GEM_HOME/bin:$PATH"

  # Unconditional -W0 to avoid printing e.g.:
  # warning: parser/current is loading parser/ruby26, which recognizes
  # warning: 2.6.6-compliant syntax, but you are running 2.6.3.
  exec "$HOMEBREW_RUBY_PATH" "$RUBY_DISABLE_OPTIONS" -W0 -S rubocop "$@"
}
