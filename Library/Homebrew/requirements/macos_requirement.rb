# frozen_string_literal: true

require "requirement"

class MacOSRequirement < Requirement
  fatal true

  def initialize(tags = [], comparator: ">=")
    @version = MacOS::Version.from_symbol(tags.shift) unless tags.empty?
    @comparator = comparator
    super(tags)
  end

  def version_specified?
    OS.mac? && @version
  end

  satisfy(build_env: false) do
    next MacOS.version.public_send(@comparator, @version) if version_specified?
    next true if OS.mac?
    next true if @version

    false
  end

  def message
    return "macOS is required." unless version_specified?

    case @comparator
    when ">="
      "macOS #{@version.pretty_name} or newer is required."
    when "<="
      <<~EOS
        This formula either does not compile or function as expected on macOS
        versions newer than #{@version.pretty_name} due to an upstream incompatibility.
      EOS
    else
      "macOS #{@version.pretty_name} is required."
    end
  end

  def display_s
    return "macOS is required" unless version_specified?

    "macOS #{@comparator} #{@version}"
  end
end
