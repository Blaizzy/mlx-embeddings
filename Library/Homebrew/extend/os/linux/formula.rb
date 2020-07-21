# frozen_string_literal: true

class Formula
  undef shared_library

  def shared_library(name, version = nil)
    "#{name}.so#{"." unless version.nil?}#{version}"
  end

  undef allowed_missing_lib?
  def allowed_missing_lib?(lib)
    raise TypeError "Library must be a string; got a #{lib.class} (#{lib})" unless lib.is_a? String

    # lib:   Full path to the missing library
    #        Ex.: /home/linuxbrew/.linuxbrew/lib/libsomething.so.1
    # x -    Name of or a pattern for a library, linkage to which is allowed to be missing.
    #        Ex. 1: "libONE.so.1"
    #        Ex. 2: %r{(libONE|libTWO)\.so}
    self.class.allowed_missing_libraries.any? do |x|
      case x
      when Regexp
        x.match? lib
      when String
        lib.include? x
      end
    end
  end

  class << self
    undef on_linux

    def on_linux(&_block)
      yield
    end

    undef ignore_missing_libraries

    def ignore_missing_libraries(*libs)
      libraries = libs.flatten
      if libraries.any? { |x| !x.is_a?(String) && !x.is_a?(Regexp) }
        raise FormulaSpecificationError, "#{__method__} can handle Strings and Regular Expressions only"
      end

      allowed_missing_libraries.merge(libraries)
    end

    # @private
    def allowed_missing_libraries
      @allowed_missing_libraries ||= Set.new
    end
  end
end
