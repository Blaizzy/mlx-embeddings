# typed: true
# frozen_string_literal: true

require "extend/os/cp"
require "fileutils"
require "system_command"

module Utils
  # Helper functions for copying files.
  module Cp
    class << self
      def with_attributes(source, target, force_command: false, sudo: false, verbose: false, command: SystemCommand)
        if force_command || sudo
          command.run! "cp", args: ["-p", *source, target], sudo:, verbose:
        else
          FileUtils.cp source, target, preserve: true, verbose:
        end

        nil
      end

      def recursive_with_attributes(source, target, force_command: false, sudo: false, verbose: false,
                                    command: SystemCommand)
        if force_command || sudo
          command.run! "cp", args: ["-pR", *source, target], sudo:, verbose:
        else
          FileUtils.cp_r source, target, preserve: true, verbose:
        end

        nil
      end
    end
  end
end
