# frozen_string_literal: true

class SoftwareSpec
  def uses_from_macos(deps, **args)
    depends_on(deps) if add_mac_dependency?(args)
  end

  private

  def add_mac_dependency?(args)
    args.each { |key, version| args[key] = OS::Mac::Version.from_symbol(version) }

    return false if args[:after] && OS::Mac.version < args[:after]

    return false if args[:before] && OS::Mac.version >= args[:before]

    true
  end
end
