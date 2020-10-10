# typed: false
# frozen_string_literal: true

class SoftwareSpec
  undef uses_from_macos

  def uses_from_macos(deps, bounds = {})
    @uses_from_macos_elements ||= []

    if deps.is_a?(Hash)
      bounds = deps.dup
      deps = Hash[*bounds.shift]
    end

    bounds = bounds.transform_values { |v| MacOS::Version.from_symbol(v) }
    if MacOS.version >= bounds[:since]
      @uses_from_macos_elements << deps
    else
      depends_on deps
    end
  end
end
