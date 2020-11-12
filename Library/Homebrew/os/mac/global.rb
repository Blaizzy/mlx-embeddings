# typed: false
# frozen_string_literal: true

module Homebrew
  DEFAULT_PREFIX ||= if Hardware::CPU.arm?
    HOMEBREW_MACOS_ARM_DEFAULT_PREFIX
  else
    HOMEBREW_DEFAULT_PREFIX
  end.freeze
end
