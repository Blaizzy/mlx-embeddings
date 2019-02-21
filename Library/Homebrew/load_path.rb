require "pathname"

HOMEBREW_LIBRARY_PATH = Pathname(__dir__).realpath

$LOAD_PATH.push(HOMEBREW_LIBRARY_PATH.to_s) unless $LOAD_PATH.include?(HOMEBREW_LIBRARY_PATH.to_s)

require "vendor/bundle-standalone/bundler/setup"
