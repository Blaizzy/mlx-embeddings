# typed: true
# frozen_string_literal: true

require "singleton"

class Version
  # A pseudo-version representing the absence of a version.
  #
  # @api private
  class NullVersion < Version
    extend T::Sig

    include Comparable
    include Singleton

    sig { override.params(_other: T.untyped).returns(Integer) }
    def <=>(_other)
      -1
    end

    sig { override.params(_other: T.untyped).returns(T::Boolean) }
    def eql?(_other)
      # Makes sure that the same instance of Version::NULL
      # will never equal itself; normally Comparable#==
      # will return true for this regardless of the return
      # value of #<=>
      false
    end

    sig { override.returns(T::Boolean) }
    def detected_from_url?
      false
    end

    sig { override.returns(T::Boolean) }
    def head?
      false
    end

    sig { override.returns(T::Boolean) }
    def null?
      true
    end

    # For {OS::Mac::Version} compatibility.
    sig { returns(T::Boolean) }
    def requires_nehalem_cpu?
      false
    end
    alias requires_sse4? requires_nehalem_cpu?
    alias requires_sse41? requires_nehalem_cpu?
    alias requires_sse42? requires_nehalem_cpu?
    alias requires_popcnt? requires_nehalem_cpu?

    sig { override.returns(Token) }
    def major
      NULL_TOKEN
    end

    sig { override.returns(Token) }
    def minor
      NULL_TOKEN
    end

    sig { override.returns(Token) }
    def patch
      NULL_TOKEN
    end

    sig { override.returns(Version) }
    def major_minor
      self
    end

    sig { override.returns(Version) }
    def major_minor_patch
      self
    end

    sig { override.returns(Float) }
    def to_f
      Float::NAN
    end

    sig { override.returns(Integer) }
    def to_i
      0
    end

    sig { override.returns(String) }
    def to_s
      ""
    end
    alias to_str to_s

    sig { override.returns(String) }
    def inspect
      "#<Version::NULL>"
    end
  end
  private_constant :NullVersion

  # Represents the absence of a version.
  NULL = NullVersion.instance.freeze
end
