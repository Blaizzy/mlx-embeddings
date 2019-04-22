# frozen_string_literal: true

require "pathname"

HOMEBREW_LIBRARY_PATH = Pathname(__dir__).realpath.freeze

$LOAD_PATH.push(HOMEBREW_LIBRARY_PATH.to_s) unless $LOAD_PATH.include?(HOMEBREW_LIBRARY_PATH.to_s)

require "vendor/bundle/bundler/setup"
