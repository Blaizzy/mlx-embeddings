# typed: true
# frozen_string_literal: true

require "simulate_system"

if OS.mac? && Hardware::CPU.arm?
  require "extend/os/mac/default_prefix"
elsif Homebrew::SimulateSystem.simulating_or_running_on_linux?
  require "extend/os/linux/default_prefix"
end
