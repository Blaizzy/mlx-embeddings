# typed: true
# frozen_string_literal: true

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
      def copy(source, target, sudo: false, verbose: false, command: SystemCommand)
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
      def copy_recursive(source, target, sudo: false, verbose: false, command: SystemCommand)
        command.run! "cp", args: ["-pR", *extra_flags, *source, target], sudo:, verbose:
      end

      private

      # Use the lightweight `clonefile(2)` syscall if applicable.
      MACOS_FLAGS = ["-c"].freeze
      GENERIC_FLAGS = [].freeze

      sig { returns(T::Array[String]) }
      def extra_flags
        # The `cp` command on older macOS versions also had the `-c` option, but before Sonoma, the
        # command would fail if the `clonefile` syscall isn't applicable (the underlying filesystem
        # doesn't support the feature or the source and the target are on different filesystems).
        if MacOS.version >= :sonoma
          MACOS_FLAGS
        else
          GENERIC_FLAGS
        end
      end
    end
  end
end
