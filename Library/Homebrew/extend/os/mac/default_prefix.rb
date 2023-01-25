# typed: true
# frozen_string_literal: true

require "simulate_system"

module Homebrew
  if Hardware::CPU.arm? || Homebrew::SimulateSystem.simulating_or_running_on_linux?
    remove_const(:DEFAULT_PREFIX)
    remove_const(:DEFAULT_REPOSITORY)

    DEFAULT_PREFIX, DEFAULT_REPOSITORY = if Hardware::CPU.arm?
      [HOMEBREW_MACOS_ARM_DEFAULT_PREFIX, HOMEBREW_MACOS_ARM_DEFAULT_REPOSITORY]
    elsif Homebrew::SimulateSystem.simulating_or_running_on_linux?
      [HOMEBREW_LINUX_DEFAULT_PREFIX, HOMEBREW_LINUX_DEFAULT_REPOSITORY]
    end
  end
end
