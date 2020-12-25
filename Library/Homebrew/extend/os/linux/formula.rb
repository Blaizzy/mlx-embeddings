# typed: true
# frozen_string_literal: true

class Formula
  undef shared_library

  def shared_library(name, version = nil)
    suffix = if version == "*" || (name == "*" && version.blank?)
      "{,.*}"
    elsif version.present?
      ".#{version}"
    end
    "#{name}.so#{suffix}"
  end

  class << self
    undef ignore_missing_libraries

    def ignore_missing_libraries(*libs)
      libraries = libs.flatten
      if libraries.any? { |x| !x.is_a?(String) && !x.is_a?(Regexp) }
        raise FormulaSpecificationError, "#{__method__} can handle Strings and Regular Expressions only"
      end

      allowed_missing_libraries.merge(libraries)
    end
  end
end
