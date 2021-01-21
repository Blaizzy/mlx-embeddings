# typed: true
# frozen_string_literal: true

require "pathname"

HOMEBREW_LIBRARY_PATH = Pathname(__dir__).realpath.freeze

$LOAD_PATH.push HOMEBREW_LIBRARY_PATH.to_s

require "vendor/bundle/bundler/setup"

if ENV["HOMEBREW_BOOTSNAP"]
  require "homebrew_bootsnap"
else
  $LOAD_PATH.select! { |d| Pathname(d).directory? }
  $LOAD_PATH.uniq!
end
