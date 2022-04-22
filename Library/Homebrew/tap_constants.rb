# typed: true
# frozen_string_literal: true

# Match taps' formulae, e.g. `someuser/sometap/someformula`
HOMEBREW_TAP_FORMULA_REGEX = %r{^([\w-]+)/([\w-]+)/([\w+-.@]+)$}.freeze
# Match taps' casks, e.g. `someuser/sometap/somecask`
HOMEBREW_TAP_CASK_REGEX = %r{^([\w-]+)/([\w-]+)/([a-z0-9\-_]+)$}.freeze
# Match taps' directory paths, e.g. `HOMEBREW_LIBRARY/Taps/someuser/sometap`
HOMEBREW_TAP_DIR_REGEX = %r{#{Regexp.escape(HOMEBREW_LIBRARY.to_s)}/Taps/(?<user>[\w-]+)/(?<repo>[\w-]+)}.freeze
# Match taps' formula paths, e.g. `HOMEBREW_LIBRARY/Taps/someuser/sometap/someformula`
HOMEBREW_TAP_PATH_REGEX = Regexp.new(HOMEBREW_TAP_DIR_REGEX.source + %r{(?:/.*)?$}.source).freeze
# Match official taps' casks, e.g. `homebrew/cask/somecask or homebrew/cask-versions/somecask`
HOMEBREW_CASK_TAP_CASK_REGEX = %r{^(?:([Cc]askroom)/(cask|versions)|(homebrew)/(cask|cask-[\w-]+))/([\w+-.]+)$}.freeze
HOMEBREW_OFFICIAL_REPO_PREFIXES_REGEX = /^(home|linux)brew-/.freeze
