# frozen_string_literal: true

require "version"

module OS
  module Mac
    class Version < ::Version
      SYMBOLS = {
        catalina:    "10.15",
        mojave:      "10.14",
        high_sierra: "10.13",
        sierra:      "10.12",
        el_capitan:  "10.11",
        yosemite:    "10.10",
        mavericks:   "10.9",
      }.freeze

      def self.from_symbol(sym)
        str = SYMBOLS.fetch(sym) do
          raise ArgumentError, "unknown version #{sym.inspect}"
        end
        new(str)
      end

      def initialize(*args)
        super
        @comparison_cache = {}
      end

      def <=>(other)
        @comparison_cache.fetch(other) do
          v = SYMBOLS.fetch(other) { other.to_s }
          @comparison_cache[other] = super(Version.new(v))
        end
      end

      def to_sym
        SYMBOLS.invert.fetch(@version) { :dunno }
      end

      def pretty_name
        to_sym.to_s.split("_").map(&:capitalize).join(" ")
      end

      # For OS::Mac::Version compatibility
      def requires_nehalem_cpu?
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
