# typed: true
# frozen_string_literal: true

require "compilers"

# Combination of C++ standard library and compiler.
class CxxStdlib
  extend T::Sig

  include CompilerConstants

  # Error for when a formula's dependency was built with a different C++ standard library.
  class CompatibilityError < StandardError
    def initialize(formula, dep, stdlib)
      super <<~EOS
        #{formula.full_name} dependency #{dep.name} was built with a different C++ standard
        library (#{stdlib.type_string} from #{stdlib.compiler}). This may cause problems at runtime.
      EOS
    end
  end

  def self.create(type, compiler)
    raise ArgumentError, "Invalid C++ stdlib type: #{type}" if type && [:libstdcxx, :libcxx].exclude?(type)

    CxxStdlib.new(type, compiler)
  end

  attr_reader :type, :compiler

  def initialize(type, compiler)
    @type = type
    @compiler = compiler.to_sym
  end

  def type_string
    type.to_s.gsub(/cxx$/, "c++")
  end

  sig { returns(String) }
  def inspect
    "#<#{self.class.name}: #{compiler} #{type}>"
  end
end
