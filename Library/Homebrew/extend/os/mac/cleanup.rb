# typed: true
# frozen_string_literal: true

module Homebrew
  class Cleanup
    undef use_system_ruby?

    def use_system_ruby?
      return false if Homebrew::EnvConfig.force_vendor_ruby?

      ENV["HOMEBREW_MACOS_SYSTEM_RUBY_NEW_ENOUGH"].present?
    end
  end
end
