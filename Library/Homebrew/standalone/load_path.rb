# typed: true
# frozen_string_literal: true

require "pathname"

HOMEBREW_LIBRARY_PATH = Pathname(__dir__).parent.realpath.freeze

require_relative "../utils/gems"
Homebrew.setup_gem_environment!(setup_path: false)

$LOAD_PATH.push HOMEBREW_LIBRARY_PATH.to_s unless $LOAD_PATH.include?(HOMEBREW_LIBRARY_PATH.to_s)
require_relative "../vendor/bundle/bundler/setup"
$LOAD_PATH.uniq!
