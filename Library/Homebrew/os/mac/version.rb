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

      sig { returns(Symbol) }
      attr_reader :arch

      SYMBOLS = {
        big_sur:     "11",
        catalina:    "10.15",
        mojave:      "10.14",
        high_sierra: "10.13",
        sierra:      "10.12",
        el_capitan:  "10.11",
        yosemite:    "10.10",
      }.freeze

      sig { params(sym: Symbol).returns(T.attached_class) }
      def self.from_symbol(sym)
        version, arch = version_arch(sym)
        version ||= sym
        str = SYMBOLS.fetch(version.to_sym) { raise MacOSVersionError, sym }
        new(str, arch: arch)
      end

      sig { params(value: T.any(String, Symbol)).returns(T.any([], [String, T.nilable(String)])) }
      def self.version_arch(value)
        @all_archs_regex ||= begin
          all_archs = Hardware::CPU::ALL_ARCHS.map(&:to_s)
          /
            ^((?<prefix_arch>#{Regexp.union(all_archs)})_)?
            (?<version>[\w.]+)
            (-(?<suffix_arch>#{Regexp.union(all_archs)}))?$
          /x
        end
        match = @all_archs_regex.match(value.to_s)
        return [] unless match

        version = match[:version]
        arch = match[:prefix_arch] || match[:suffix_arch]
        [version, arch]
      end

      sig { params(value: T.nilable(String), arch: T.nilable(String)).void }
      def initialize(value, arch: nil)
        version, arch = Version.version_arch(value) if value.present? && arch.nil?
        version ||= value
        arch    ||= "intel"

        raise MacOSVersionError, version unless /\A1\d+(?:\.\d+){0,2}\Z/.match?(version)

        super(version)

        @arch = arch.to_sym
        @comparison_cache = {}
      end

      sig { override.params(other: T.untyped).returns(T.nilable(Integer)) }
      def <=>(other)
        @comparison_cache.fetch(other) do
          if SYMBOLS.key?(other) && to_sym == other
            0
          else
            v = SYMBOLS.fetch(other) { other.to_s }
            @comparison_cache[other] = super(::Version.new(v))
          end
        end
      end

      sig { returns(Symbol) }
      def to_sym
        @to_sym ||= begin
          # Big Sur is 11.x but Catalina is 10.15.
          major_macos = if major >= 11
            major
          else
            major_minor
          end.to_s
          SYMBOLS.invert.fetch(major_macos, :dunno)
        end
      end

      sig { returns(String) }
      def pretty_name
        @pretty_name ||= to_sym.to_s.split("_").map(&:capitalize).join(" ").freeze
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
