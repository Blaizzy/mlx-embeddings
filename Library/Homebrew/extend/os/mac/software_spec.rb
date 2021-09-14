# typed: true
# frozen_string_literal: true

# The Library/Homebrew/extend/os/software_spec.rb conditional logic will need to be more nuanced
# if this file ever includes more than `uses_from_macos`.
class SoftwareSpec
  undef uses_from_macos

  def uses_from_macos(deps, bounds = {})
    if deps.is_a?(Hash)
      bounds = deps.dup
      deps = [bounds.shift].to_h
    end

    @uses_from_macos_elements << deps

    bounds = bounds.transform_values { |v| MacOS::Version.from_symbol(v) }

    # Linux simulating macOS. Assume oldest macOS version.
    return if Homebrew::EnvConfig.simulate_macos_on_linux? && !bounds.key?(:since)

    # macOS new enough for dependency to not be required.
    return if MacOS.version >= bounds[:since]

    depends_on deps
  end
end
