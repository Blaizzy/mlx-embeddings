# frozen_string_literal: true

class Version
  NULL = Class.new do
    include Comparable

    def <=>(_other)
      -1
    end

    def eql?(_other)
      # Makes sure that the same instance of Version::NULL
      # will never equal itself; normally Comparable#==
      # will return true for this regardless of the return
      # value of #<=>
      false
    end

    def detected_from_url?
      false
    end

    def head?
      false
    end

    def null?
      true
    end

    # For OS::Mac::Version compatibility
    def requires_nehalem_cpu?
      false
    end
    alias_method :requires_sse4?, :requires_nehalem_cpu?
    alias_method :requires_sse41?, :requires_nehalem_cpu?
    alias_method :requires_sse42?, :requires_nehalem_cpu?
    alias_method :requires_popcnt?, :requires_nehalem_cpu?

    def to_f
      Float::NAN
    end

    def to_i
      0
    end

    def to_s
      ""
    end
    alias_method :to_str, :to_s

    def inspect
      "#<Version::NULL>"
    end
  end.new.freeze
end
