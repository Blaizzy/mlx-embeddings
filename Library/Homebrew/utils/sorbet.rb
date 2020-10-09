# typed: strict
# frozen_string_literal: true

if ENV["HOMEBREW_SORBET_RUNTIME"]
  require "utils/gems"
  Homebrew.install_bundler_gems!
  require "sorbet-runtime"
else
  begin
    gem "sorbet-runtime"
    raise "Loaded `sorbet-runtime` instead of `sorbet-runtime-stub`."
  rescue Gem::LoadError
    nil
  end

  require "sorbet-runtime-stub"
end
