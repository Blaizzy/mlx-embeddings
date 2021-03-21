# typed: strict
# frozen_string_literal: true

# This logic will need to be more nuanced if this file includes more than `uses_from_macos`.
if OS.mac? || Homebrew::EnvConfig.simulate_macos_on_linux?
  require "extend/os/mac/software_spec"
elsif OS.linux?
  require "extend/os/linux/software_spec"
end
