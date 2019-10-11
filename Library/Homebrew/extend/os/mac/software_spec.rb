# frozen_string_literal: true

class SoftwareSpec
  undef uses_from_macos

  def uses_from_macos(deps, **args)
    @uses_from_macos_elements ||= []

    if deps.is_a?(Hash)
      args = deps
      deps = Hash[*args.shift]
    end

    if add_mac_dependency?(args)
      depends_on(deps)
    else
      @uses_from_macos_elements << deps
    end
  end

  private

  def add_mac_dependency?(args)
    args.each { |key, version| args[key] = OS::Mac::Version.from_symbol(version) }

    return false if args[:after] && OS::Mac.version >= args[:after]

    return false if args[:before] && OS::Mac.version < args[:before]

    args.present?
  end
end
