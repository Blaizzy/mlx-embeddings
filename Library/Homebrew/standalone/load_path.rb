# typed: true
# frozen_string_literal: true

# We trust base Ruby to provide what we need.
# Don't look into the user-installed sitedir, which may contain older versions of RubyGems.
require "rbconfig"
$LOAD_PATH.reject! { |path| path.start_with?(RbConfig::CONFIG["sitedir"]) }

require "pathname"
HOMEBREW_LIBRARY_PATH = Pathname(__dir__).parent.realpath.freeze

require_relative "../utils/gems"
Homebrew.setup_gem_environment!(setup_path: false)

$LOAD_PATH.push HOMEBREW_LIBRARY_PATH.to_s unless $LOAD_PATH.include?(HOMEBREW_LIBRARY_PATH.to_s)
require_relative "../vendor/bundle/bundler/setup"
$LOAD_PATH.unshift "#{HOMEBREW_LIBRARY_PATH}/vendor/bundle/#{RUBY_ENGINE}/#{Gem.ruby_api_version}/gems/" \
                   "bundler-#{Homebrew::HOMEBREW_BUNDLER_VERSION}/lib"
$LOAD_PATH.uniq!
