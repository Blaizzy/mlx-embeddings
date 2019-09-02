# frozen_string_literal: true

require "requirement"

class MacOSRequirement < Requirement
  fatal true

  def initialize(tags = [], comparator: ">=")
    if comparator == "==" && tags.first.respond_to?(:map)
      @version = tags.shift.map { |s| MacOS::Version.from_symbol(s) }
    else
      @version = MacOS::Version.from_symbol(tags.shift) unless tags.empty?
    end

    @comparator = comparator
    super(tags)
  end

  def version_specified?
    OS.mac? && @version
  end

  satisfy(build_env: false) do
    next [*@version].any? { |v| MacOS.version.public_send(@comparator, v) } if version_specified?
    next true if OS.mac?
    next true if @version

    false
  end

  def message(type: :formula)
    return "macOS is required." unless version_specified?

    case @comparator
    when ">="
      "macOS #{@version.pretty_name} or newer is required."
    when "<="
      case type
      when :formula
        <<~EOS
          This formula either does not compile or function as expected on macOS
          versions newer than #{@version.pretty_name} due to an upstream incompatibility.
        EOS
      when :cask
        "This cask does not run on macOS versions newer than #{@version.pretty_name}."
      end
    else
      if @version.respond_to?(:to_ary)
        *versions, last = @version.map(&:pretty_name)
        return "macOS #{versions.join(", ")} or #{last} is required."
      end

      "macOS #{@version.pretty_name} is required."
    end
  end

  def display_s
    return "macOS is required" unless version_specified?

    "macOS #{@comparator} #{@version}"
  end

  def to_json(*args)
    comp = @comparator.to_s
    return { comp => @version.map(&:to_s) }.to_json(*args) if @version.is_a?(Array)

    { comp => [@version.to_s] }.to_json(*args)
  end
end
