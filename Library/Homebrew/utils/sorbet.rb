# typed: strict
# frozen_string_literal: true

if ENV["HOMEBREW_SORBET_RUNTIME"]
  require "utils/gems"
  Homebrew.install_bundler_gems!
  require "sorbet-runtime"
else
  require "sorbet-runtime-stub"
end
