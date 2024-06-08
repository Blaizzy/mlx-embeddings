# typed: true
# frozen_string_literal: true

require "extend/os/cp"
require "fileutils"
require "system_command"

module Utils
  # Helper functions for copying files.
  module Cp
    class << self
      def with_attributes(source, target, sudo: false, verbose: false, command: SystemCommand)
        odisabled "`Utils::Cp.with_attributes` with `sudo: true` on Linux" if sudo
        FileUtils.cp source, target, preserve: true, verbose:
      end

      def recursive_with_attributes(source, target, sudo: false, verbose: false, command: SystemCommand)
        odisabled "`Utils::Cp.recursive_with_attributes` with `sudo: true` on Linux" if sudo
        FileUtils.cp_r source, target, preserve: true, verbose:
      end
    end
  end
end
