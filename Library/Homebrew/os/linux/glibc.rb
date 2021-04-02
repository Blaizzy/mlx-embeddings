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
        @system_version ||= begin
          version = Utils.popen_read("/usr/bin/ldd", "--version")[/ (\d+\.\d+)/, 1]
          if version
            Version.new version
          else
            Version::NULL
          end
        end
      end

      def version
        @version ||= begin
          version = Utils.popen_read(HOMEBREW_PREFIX/"opt/glibc/bin/ldd", "--version")[/ (\d+\.\d+)/, 1]
          if version
            Version.new version
          else
            system_version
          end
        end
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
