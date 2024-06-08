# typed: true
# frozen_string_literal: true

module Utils
  module Cp
    class << self
      module MacOSOverride
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
          if (flags = extra_flags)
            command.run! "cp", args: ["-p", *flags, *source, target], sudo:, verbose:
            nil
          else
            super
          end
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
          if (flags = extra_flags)
            command.run! "cp", args: ["-pR", *flags, *source, target], sudo:, verbose:
            nil
          else
            super
          end
        end

        private

        # Use the lightweight `clonefile(2)` syscall if applicable.
        SONOMA_FLAGS = ["-c"].freeze

        sig { returns(T.nilable(T::Array[String])) }
        def extra_flags
          # The `cp` command on older macOS versions also had the `-c` option, but before Sonoma,
          # the command would fail if the `clonefile` syscall isn't applicable (the underlying
          # filesystem doesn't support the feature or the source and the target are on different
          # filesystems).
          return if MacOS.version < :sonoma

          SONOMA_FLAGS
        end
      end

      prepend MacOSOverride
    end
  end
end
