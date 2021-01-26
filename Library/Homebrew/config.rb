# typed: true
# frozen_string_literal: true

raise "HOMEBREW_BREW_FILE was not exported! Please call bin/brew directly!" unless ENV["HOMEBREW_BREW_FILE"]

# Path to `bin/brew` main executable in `HOMEBREW_PREFIX`
HOMEBREW_BREW_FILE = Pathname(ENV["HOMEBREW_BREW_FILE"]).freeze

class MissingEnvironmentVariables < RuntimeError; end

# Helper module for getting environment variables which must be set.
#
# @api private
module EnvVar
  extend T::Sig

  sig { params(env: String).returns(String) }
  def self.[](env)
    raise MissingEnvironmentVariables, "#{env} was not exported!" unless ENV[env]

    ENV.fetch(env)
  end
end

require "extend/git_repository"

# Where we link under
HOMEBREW_PREFIX = Pathname(EnvVar["HOMEBREW_PREFIX"]).freeze

# Where `.git` is found
HOMEBREW_REPOSITORY = Pathname(EnvVar["HOMEBREW_REPOSITORY"])
                      .extend(GitRepositoryExtension)
                      .freeze

# Where we store most of Homebrew, taps, and various metadata
HOMEBREW_LIBRARY = Pathname(EnvVar["HOMEBREW_LIBRARY"]).freeze

# Where shim scripts for various build and SCM tools are stored
HOMEBREW_SHIMS_PATH = (HOMEBREW_LIBRARY/"Homebrew/shims").freeze

# Where external data that has been incorporated into Homebrew is stored
HOMEBREW_DATA_PATH = (HOMEBREW_LIBRARY/"Homebrew/data").freeze

# Where we store symlinks to currently linked kegs
HOMEBREW_LINKED_KEGS = (HOMEBREW_PREFIX/"var/homebrew/linked").freeze

# Where we store symlinks to currently version-pinned kegs
HOMEBREW_PINNED_KEGS = (HOMEBREW_PREFIX/"var/homebrew/pinned").freeze

# Where we store lock files
HOMEBREW_LOCKS = (HOMEBREW_PREFIX/"var/homebrew/locks").freeze

# Where we store built products
HOMEBREW_CELLAR = Pathname(EnvVar["HOMEBREW_CELLAR"]).freeze

# Where downloads (bottles, source tarballs, etc.) are cached
HOMEBREW_CACHE = Pathname(EnvVar["HOMEBREW_CACHE"]).freeze

# Where formulae installed via URL are cached
HOMEBREW_CACHE_FORMULA = (HOMEBREW_CACHE/"Formula").freeze

# Where build, postinstall, and test logs of formulae are written to
HOMEBREW_LOGS = Pathname(EnvVar["HOMEBREW_LOGS"]).expand_path.freeze

# Must use `/tmp` instead of `TMPDIR` because long paths break Unix domain sockets
HOMEBREW_TEMP = Pathname(EnvVar["HOMEBREW_TEMP"]).yield_self do |tmp|
  tmp.mkpath unless tmp.exist?
  tmp.realpath
end.freeze
