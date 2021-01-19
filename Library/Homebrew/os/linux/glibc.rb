# typed: true
# frozen_string_literal: true

module OS
  module Linux
    # Helper functions for querying `glibc` information.
    #
    # @api private
    module Glibc
      extend T::Sig

      module_function

      def system_version
        return @system_version if @system_version

        version = Utils.popen_read("/usr/bin/ldd", "--version")[/ (\d+\.\d+)/, 1]
        return Version::NULL unless version

        @system_version = Version.new version
      end

      sig { returns(Version) }
      def minimum_version
        Version.new(ENV.fetch("HOMEBREW_LINUX_MINIMUM_GLIBC_VERSION"))
      end

      def below_minimum_version?
        system_version < minimum_version
      end
    end
  end
end
