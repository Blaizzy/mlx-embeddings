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
        command.run! executable, args: ["-p", *extra_flags, *source, target], sudo:, verbose:
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
        command.run! executable, args: ["-pR", *extra_flags, *source, target], sudo:, verbose:
      end

      private

      GCP = (HOMEBREW_PREFIX/"opt/coreutils/libexec/gnubin/cp").freeze
      UCP = (HOMEBREW_PREFIX/"opt/uutils-coreutils/libexec/uubin/cp").freeze

      sig { returns(T.any(String, Pathname)) }
      def executable
        case type
        when :macos
          Pathname("/bin/cp")
        when :coreutils
          GCP
        when :uutils
          UCP
        else
          "cp"
        end
      end

      MACOS_FLAGS = [
        # Perform a lightweight copy-on-write clone if applicable.
        "-c",
      ].freeze
      GNU_FLAGS = [
        # Unlike BSD cp, `gcp -p` doesn't guarantee to preserve extended attributes, including
        # quarantine information on macOS.
        "--preserve=all",
        "--no-preserve=links",
        # Equivalent to `-c` on macOS.
        "--reflink=auto",
      ].freeze
      GENERIC_FLAGS = [].freeze

      sig { returns(T::Array[String]) }
      def extra_flags
        case type
        when :macos
          MACOS_FLAGS
        when :coreutils, :uutils
          GNU_FLAGS
        else
          GENERIC_FLAGS
        end
      end

      sig { returns(T.nilable(Symbol)) }
      def type
        return @type if defined?(@type)

        # The `cp` command on some older macOS versions also had the `-c` option, but before Sonoma,
        # the command would fail if the `clonefile` syscall isn't applicable (the underlying
        # filesystem doesn't support the feature or the source and the target are on different
        # filesystems).
        return @type = :macos if MacOS.version >= :sonoma

        {
          coreutils: "coreutils",
          uutils:    "uutils-coreutils",
        }.each do |type, formula|
          begin
            formula = Formula[formula]
          rescue FormulaUnavailableError
            next
          end
          return @type = type if formula.optlinked?
        end

        @type = nil
      end
    end
  end
end
