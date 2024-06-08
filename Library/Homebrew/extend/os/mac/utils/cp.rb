# typed: true
# frozen_string_literal: true

module Utils
  module Cp
    class << self
      module MacOSOverride
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
          # `cp -p` on macOS guarantees to preserve extended attributes (including quarantine
          # information) in addition to file mode, which is requered when copying cask artifacts.
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

        # Use the lightweight `clonefile(2)` syscall if applicable.
        SONOMA_FLAGS = ["-c"].freeze

        sig { returns(T::Array[String]) }
        def extra_flags
          # The `cp` command on older macOS versions also had the `-c` option, but before Sonoma,
          # the command would fail if the `clonefile` syscall isn't applicable (the underlying
          # filesystem doesn't support the feature or the source and the target are on different
          # filesystems).
          if MacOS.version >= :sonoma
            SONOMA_FLAGS
          else
            [].freeze
          end
        end
      end

      prepend MacOSOverride
    end
  end
end
