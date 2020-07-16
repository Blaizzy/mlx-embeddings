# frozen_string_literal: true

class Formula
  undef shared_library

  def shared_library(name, version = nil)
    "#{name}.so#{"." unless version.nil?}#{version}"
  end

  undef allowed_missing_lib?
  def allowed_missing_lib?(lib)
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

    def ignore_missing_libraries(*libs)
      libs.flatten!
      allowed_missing_libraries.merge(libs)
    end

    # @private
    def allowed_missing_libraries
      @allowed_missing_libraries ||= Set.new
    end
  end
end
