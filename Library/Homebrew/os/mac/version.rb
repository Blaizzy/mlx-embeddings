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
        return @comparison_cache[other] if @comparison_cache.key?(other)

        result = case other
        when Symbol
          if MacOSVersions::SYMBOLS.key?(other) && to_sym == other
            0
          else
            v = MacOSVersions::SYMBOLS.fetch(other) { other.to_s }
            super(v)
          end
        else
          super
        end

        @comparison_cache[other] = result unless frozen?

        result
      end

      sig { returns(T.self_type) }
      def strip_patch
        return self if null?

        # Big Sur is 11.x but Catalina is 10.15.x.
        if T.must(major) >= 11
          self.class.new(major.to_s)
        else
          major_minor
        end
      end

      sig { returns(Symbol) }
      def to_sym
        return @sym if defined?(@sym)

        sym = MacOSVersions::SYMBOLS.invert.fetch(strip_patch.to_s, :dunno)

        @sym = sym unless frozen?

        sym
      end

      sig { returns(String) }
      def pretty_name
        return @pretty_name if defined?(@pretty_name)

        pretty_name = to_sym.to_s.split("_").map(&:capitalize).join(" ").freeze

        @pretty_name = pretty_name unless frozen?

        pretty_name
      end

      sig { returns(T::Boolean) }
      def outdated_release?
        self < HOMEBREW_MACOS_OLDEST_SUPPORTED
      end

      sig { returns(T::Boolean) }
      def prerelease?
        self >= HOMEBREW_MACOS_NEWEST_UNSUPPORTED
      end

      sig { returns(T::Boolean) }
      def unsupported_release?
        outdated_release? || prerelease?
      end

      # For {OS::Mac::Version} compatibility.
      sig { returns(T::Boolean) }
      def requires_nehalem_cpu?
        return false if null?

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

      # Represents the absence of a version.
      # NOTE: Constructor needs to called with an arbitrary macOS-like version which is then set to `nil`.
      NULL = Version.new("10.0").tap { |v| v.instance_variable_set(:@version, nil) }.freeze
    end
  end
end
