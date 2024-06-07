# typed: true
# frozen_string_literal: true

module Utils
  module Cp
    class << self
      module MacOSOverride
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
            super
          end
        end
      end

      prepend MacOSOverride
    end
  end
end
