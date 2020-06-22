# frozen_string_literal: true

# enables experimental readelf.rb, patchelf support.
HOMEBREW_PATCHELF_RB = (ENV["HOMEBREW_PATCHELF_RB"].present? ||
                        (ENV["HOMEBREW_DEVELOPER"].present? && ENV["HOMEBREW_NO_PATCHELF_RB"].blank?)).freeze

module Homebrew
  DEFAULT_PREFIX ||= if Homebrew::EnvConfig.force_homebrew_on_linux?
    HOMEBREW_DEFAULT_PREFIX
  else
    LINUXBREW_DEFAULT_PREFIX
  end.freeze
end
