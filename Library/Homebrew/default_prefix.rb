# typed: true
# frozen_string_literal: true

require "simulate_system"

module Homebrew
  # TODO: Refactor and move to extend/os
  DEFAULT_PREFIX, DEFAULT_REPOSITORY = if OS.mac? && Hardware::CPU.arm? # rubocop:disable Homebrew/MoveToExtendOS
    [HOMEBREW_MACOS_ARM_DEFAULT_PREFIX, HOMEBREW_MACOS_ARM_DEFAULT_REPOSITORY]
  elsif Homebrew::SimulateSystem.simulating_or_running_on_linux?
    [HOMEBREW_LINUX_DEFAULT_PREFIX, HOMEBREW_LINUX_DEFAULT_REPOSITORY]
  else
    [HOMEBREW_DEFAULT_PREFIX, HOMEBREW_DEFAULT_REPOSITORY]
  end.freeze
end
