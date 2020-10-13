# typed: strict
# frozen_string_literal: true

if ENV["HOMEBREW_SORBET_RUNTIME"]
  require "utils/gems"
  Homebrew.install_bundler_gems!
  require "sorbet-runtime"
else
  # Explicitly prevent `sorbet-runtime` from being loaded.
  ENV["GEM_SKIP"] = "sorbet-runtime"

  require "sorbet-runtime-stub"
end
