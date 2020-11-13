# typed: true
# frozen_string_literal: true

if ENV["HOMEBREW_SORBET_RUNTIME"]
  require "utils/gems"
  Homebrew.install_bundler_gems!
  require "sorbet-runtime"
else
  # Explicitly prevent `sorbet-runtime` from being loaded.
  def gem(name, *)
    raise Gem::LoadError if name == "sorbet-runtime"

    super
  end

  require "sorbet-runtime-stub"
end
