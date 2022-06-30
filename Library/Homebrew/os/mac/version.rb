# typed: true
# frozen_string_literal: true

require "exceptions"
require "hardware"
require "version"

module OS
  module Mac
    # A macOS version.
    #
    # @api private
    class Version < ::Version
      extend T::Sig

      # TODO: bump version when new macOS is released or announced
      # and also update references in docs/Installation.md,
      # https://github.com/Homebrew/install/blob/HEAD/install.sh and
      # MacOSVersions::SYMBOLS
      NEWEST_UNSUPPORTED = "13"
      private_constant :NEWEST_UNSUPPORTED

      # TODO: bump version when new macOS is released and also update
      # references in docs/Installation.md and
      # https://github.com/Homebrew/install/blob/HEAD/install.sh
      OLDEST_SUPPORTED = "10.15"
      private_constant :OLDEST_SUPPORTED

      OLDEST_ALLOWED = "10.11"

      sig { params(version: Symbol).returns(T.attached_class) }
      def self.from_symbol(version)
        str = MacOSVersions::SYMBOLS.fetch(version) { raise MacOSVersionError, version }
        new(str)
      end

      sig { params(value: T.nilable(String)).void }
      def initialize(value)
        version ||= value

        raise MacOSVersionError, version unless /\A1\d+(?:\.\d+){0,2}\Z/.match?(version)

        super(version)

        @comparison_cache = {}
      end

      sig { override.params(other: T.untyped).returns(T.nilable(Integer)) }
      def <=>(other)
        @comparison_cache.fetch(other) do
          if MacOSVersions::SYMBOLS.key?(other) && to_sym == other
            0
          else
            v = MacOSVersions::SYMBOLS.fetch(other) { other.to_s }
            @comparison_cache[other] = super(::Version.new(v))
          end
        end
      end

      sig { returns(T.self_type) }
      def strip_patch
        # Big Sur is 11.x but Catalina is 10.15.x.
        if major >= 11
          self.class.new(major.to_s)
        else
          major_minor
        end
      end

      sig { returns(Symbol) }
      def to_sym
        @to_sym ||= MacOSVersions::SYMBOLS.invert.fetch(strip_patch.to_s, :dunno)
      end

      sig { returns(String) }
      def pretty_name
        @pretty_name ||= to_sym.to_s.split("_").map(&:capitalize).join(" ").freeze
      end

      sig { returns(T::Boolean) }
      def outdated_release?
        self < OLDEST_SUPPORTED
      end

      sig { returns(T::Boolean) }
      def prerelease?
        self >= NEWEST_UNSUPPORTED
      end

      # For {OS::Mac::Version} compatibility.
      sig { returns(T::Boolean) }
      def requires_nehalem_cpu?
        unless Hardware::CPU.intel?
          raise "Unexpected architecture: #{Hardware::CPU.arch}. This only works with Intel architecture."
        end

        Hardware.oldest_cpu(self) == :nehalem
      end
      # https://en.wikipedia.org/wiki/Nehalem_(microarchitecture)
      # Ensure any extra methods are also added to version/null.rb
      alias requires_sse4? requires_nehalem_cpu?
      alias requires_sse41? requires_nehalem_cpu?
      alias requires_sse42? requires_nehalem_cpu?
      alias requires_popcnt? requires_nehalem_cpu?
    end
  end
end
