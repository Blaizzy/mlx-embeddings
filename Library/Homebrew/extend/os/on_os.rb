# typed: strict
# frozen_string_literal: true

if OS.mac? || Homebrew::EnvConfig.simulate_macos_on_linux?
  require "extend/os/mac/on_os"
elsif OS.linux?
  require "extend/os/linux/on_os"
end
