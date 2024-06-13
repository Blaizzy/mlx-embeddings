# typed: true
# frozen_string_literal: true

require "extend/os/copy"
require "fileutils"
require "system_command"

module Utils
  # Helper functions for copying files.
  module Copy
    class << self
      sig {
        params(
          source:        T.any(String, Pathname, T::Array[T.any(String, Pathname)]),
          target:        T.any(String, Pathname),
          force_command: T::Boolean,
          sudo:          T::Boolean,
          verbose:       T::Boolean,
          command:       T.class_of(SystemCommand),
        ).void
      }
      def with_attributes(source, target, force_command: false, sudo: false, verbose: false, command: SystemCommand)
        if force_command || sudo || (flags = extra_flags)
          command.run! "cp", args: ["-p", *flags, *source, target], sudo:, verbose:
        else
          FileUtils.cp source, target, preserve: true, verbose:
        end

        nil
      end

      sig {
        params(
          source:        T.any(String, Pathname, T::Array[T.any(String, Pathname)]),
          target:        T.any(String, Pathname),
          force_command: T::Boolean,
          sudo:          T::Boolean,
          verbose:       T::Boolean,
          command:       T.class_of(SystemCommand),
        ).void
      }
      def recursive_with_attributes(source, target, force_command: false, sudo: false, verbose: false,
                                    command: SystemCommand)
        if force_command || sudo || (flags = extra_flags)
          command.run! "cp", args: ["-pR", *flags, *source, target], sudo:, verbose:
        else
          FileUtils.cp_r source, target, preserve: true, verbose:
        end

        nil
      end

      private

      def extra_flags; end
    end
  end
end
