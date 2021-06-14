# typed: true
# frozen_string_literal: true

if ENV["HOMEBREW_SORBET_RUNTIME"]
  # This is only supported under the brew environment.
  Homebrew.install_bundler_gems!(groups: ["sorbet"])
  require "sorbet-runtime"
else
  require "standalone/sorbet"
end
