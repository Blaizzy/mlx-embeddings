# typed: false
# frozen_string_literal: true

module Homebrew
  if Hardware::CPU.arm?
    DEFAULT_PREFIX ||= HOMEBREW_MACOS_ARM_DEFAULT_PREFIX.freeze
    DEFAULT_REPOSITORY ||= HOMEBREW_MACOS_ARM_DEFAULT_REPOSITORY.freeze
  else
    DEFAULT_PREFIX ||= HOMEBREW_DEFAULT_PREFIX.freeze
    DEFAULT_REPOSITORY ||= HOMEBREW_DEFAULT_REPOSITORY.freeze
  end
end
