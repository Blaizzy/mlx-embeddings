# typed: true
# frozen_string_literal: true

class Version
  # Represents the absence of a version.
  NULL = Class.new do
    extend T::Sig

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

    sig { returns(T::Boolean) }
    def detected_from_url?
      false
    end

    sig { returns(T::Boolean) }
    def head?
      false
    end

    sig { returns(T::Boolean) }
    def null?
      true
    end

    # For {OS::Mac::Version} compatibility.
    sig { returns(T::Boolean) }
    def requires_nehalem_cpu?
      false
    end
    alias_method :requires_sse4?, :requires_nehalem_cpu?
    alias_method :requires_sse41?, :requires_nehalem_cpu?
    alias_method :requires_sse42?, :requires_nehalem_cpu?
    alias_method :requires_popcnt?, :requires_nehalem_cpu?

    def major
      NULL_TOKEN
    end

    def minor
      NULL_TOKEN
    end

    def patch
      NULL_TOKEN
    end

    sig { returns(Version) }
    def major_minor
      self
    end

    sig { returns(Version) }
    def major_minor_patch
      self
    end

    sig { returns(Float) }
    def to_f
      Float::NAN
    end

    sig { returns(Integer) }
    def to_i
      0
    end

    sig { returns(String) }
    def to_s
      ""
    end
    alias_method :to_str, :to_s

    sig { returns(String) }
    def inspect
      "#<Version::NULL>"
    end
  end.new.freeze
end
