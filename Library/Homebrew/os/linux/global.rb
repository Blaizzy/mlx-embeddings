# frozen_string_literal: true

# enables experimental patchelf.rb write support.
HOMEBREW_PATCHELF_RB_WRITE = (
  ENV["HOMEBREW_NO_PATCHELF_RB_WRITE"].blank? &&
  (ENV["HOMEBREW_PATCHELF_RB_WRITE"].present? ||
   (ENV["CI"].blank? && ENV["HOMEBREW_DEVELOPER"].present?))
).freeze

module Homebrew
  DEFAULT_PREFIX ||= if Homebrew::EnvConfig.force_homebrew_on_linux?
    HOMEBREW_DEFAULT_PREFIX
  else
    LINUXBREW_DEFAULT_PREFIX
  end.freeze
end
