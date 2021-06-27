# typed: true
# frozen_string_literal: true

module Homebrew
  DEFAULT_PREFIX, DEFAULT_REPOSITORY = if OS.mac? && Hardware::CPU.arm?
    [HOMEBREW_MACOS_ARM_DEFAULT_PREFIX, HOMEBREW_MACOS_ARM_DEFAULT_REPOSITORY]
  elsif OS.linux? && !EnvConfig.simulate_macos_on_linux?
    [HOMEBREW_LINUX_DEFAULT_PREFIX, HOMEBREW_LINUX_DEFAULT_REPOSITORY]
  else
    [HOMEBREW_DEFAULT_PREFIX, HOMEBREW_DEFAULT_REPOSITORY]
  end.freeze
end
