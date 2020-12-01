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

    apple_compiler = compiler.to_s.match?(GNU_GCC_REGEXP) ? false : true
    CxxStdlib.new(type, compiler, apple_compiler)
  end

  def self.check_compatibility(formula, deps, keg, compiler)
    return if formula.skip_cxxstdlib_check?

    stdlib = create(keg.detect_cxx_stdlibs.first, compiler)

    begin
      stdlib.check_dependencies(formula, deps)
    rescue CompatibilityError => e
      opoo e.message
    end
  end

  attr_reader :type, :compiler

  def initialize(type, compiler, apple_compiler)
    @type = type
    @compiler = compiler.to_sym
    @apple_compiler = apple_compiler
  end

  # If either package doesn't use C++, all is well.
  # libstdc++ and libc++ aren't ever intercompatible.
  # libstdc++ is compatible across Apple compilers, but
  # not between Apple and GNU compilers, nor between GNU compiler versions.
  def compatible_with?(other)
    return true if type.nil? || other.type.nil?

    return false unless type == other.type

    apple_compiler? && other.apple_compiler? ||
      !other.apple_compiler? && compiler.to_s[4..6] == other.compiler.to_s[4..6]
  end

  def check_dependencies(formula, deps)
    deps.each do |dep|
      # Software is unlikely to link against libraries from build-time deps, so
      # it doesn't matter if they link against different C++ stdlibs.
      next if dep.build?

      dep_stdlib = Tab.for_formula(dep.to_formula).cxxstdlib
      raise CompatibilityError.new(formula, dep, dep_stdlib) unless compatible_with? dep_stdlib
    end
  end

  def type_string
    type.to_s.gsub(/cxx$/, "c++")
  end

  sig { returns(String) }
  def inspect
    "#<#{self.class.name}: #{compiler} #{type}>"
  end

  def apple_compiler?
    @apple_compiler
  end
end
