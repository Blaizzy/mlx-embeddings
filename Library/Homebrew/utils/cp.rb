# typed: true
# frozen_string_literal: true

require "extend/os/cp"
require "system_command"

module Utils
  # Helper functions for interacting with the `cp` command.
  module Cp
    class << self
      sig {
        params(
          source:  T.any(String, Pathname, T::Array[T.any(String, Pathname)]),
          target:  T.any(String, Pathname),
          sudo:    T::Boolean,
          verbose: T::Boolean,
          command: T.class_of(SystemCommand),
        ).returns(SystemCommand::Result)
      }
      def with_attributes(source, target, sudo: false, verbose: false, command: SystemCommand)
        # On macOS, `cp -p` guarantees to preserve extended attributes (including quarantine
        # information) in addition to file mode. Other implementations like coreutils does not
        # necessarily guarantee the same behavior, but that is fine because we don't really need to
        # preserve extended attributes except when copying Cask artifacts.
        command.run! "cp", args: ["-p", *extra_flags, *source, target], sudo:, verbose:
      end

      sig {
        params(
          source:  T.any(String, Pathname, T::Array[T.any(String, Pathname)]),
          target:  T.any(String, Pathname),
          sudo:    T::Boolean,
          verbose: T::Boolean,
          command: T.class_of(SystemCommand),
        ).returns(SystemCommand::Result)
      }
      def recursive_with_attributes(source, target, sudo: false, verbose: false, command: SystemCommand)
        command.run! "cp", args: ["-pR", *extra_flags, *source, target], sudo:, verbose:
      end

      private

      GENERIC_FLAGS = [].freeze

      def extra_flags
        GENERIC_FLAGS
      end
    end
  end
end
