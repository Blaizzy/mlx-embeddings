# typed: false
# frozen_string_literal: true

require "requirement"

# A requirement on macOS.
#
# @api private
class MacOSRequirement < Requirement
  extend T::Sig

  fatal true

  attr_reader :comparator, :version

  # TODO: when Yosemite is removed here, keep these around as empty arrays so we
  # can keep the deprecation/disabling code the same.
  DISABLED_MACOS_VERSIONS = [].freeze
  DEPRECATED_MACOS_VERSIONS = [
    :yosemite,
  ].freeze

  def initialize(tags = [], comparator: ">=")
    @version = begin
      if comparator == "==" && tags.first.respond_to?(:map)
        tags.first.map { |s| MacOS::Version.from_symbol(s) }
      else
        MacOS::Version.from_symbol(tags.first) unless tags.empty?
      end
    rescue MacOSVersionError => e
      if DISABLED_MACOS_VERSIONS.include?(e.version)
        odisabled "depends_on :macos => :#{e.version}"
      elsif DEPRECATED_MACOS_VERSIONS.include?(e.version)
        odeprecated "depends_on :macos => :#{e.version}"
      else
        raise
      end

      # Array of versions: remove the bad ones and try again.
      if tags.first.respond_to?(:reject)
        tags = [tags.first.reject { |s| s == e.version }, tags[1..]]
        retry
      end

      # Otherwise fallback to the oldest allowed if comparator is >=.
      MacOS::Version.new(MacOS::Version::OLDEST_ALLOWED) if comparator == ">="
    end

    @comparator = comparator
    super(tags.drop(1))
  end

  def version_specified?
    OS.mac? && @version
  end

  satisfy(build_env: false) do
    next Array(@version).any? { |v| MacOS.version.public_send(@comparator, v) } if version_specified?
    next true if OS.mac?
    next true if @version

    false
  end

  def message(type: :formula)
    return "macOS is required for this software." unless version_specified?

    case @comparator
    when ">="
      "macOS #{@version.pretty_name} or newer is required for this software."
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
        return "macOS #{versions.join(", ")} or #{last} is required for this software."
      end

      "macOS #{@version.pretty_name} is required for this software."
    end
  end

  sig { returns(String) }
  def inspect
    "#<#{self.class.name}: version#{@comparator}#{@version.to_s.inspect} #{tags.inspect}>"
  end

  sig { returns(String) }
  def display_s
    if version_specified?
      if @version.respond_to?(:to_ary)
        "macOS #{@comparator} #{version.join(" / ")}"
      else
        "macOS #{@comparator} #{@version}"
      end
    else
      "macOS"
    end
  end

  def to_json(*args)
    comp = @comparator.to_s
    return { comp => @version.map(&:to_s) }.to_json(*args) if @version.is_a?(Array)

    { comp => [@version.to_s] }.to_json(*args)
  end
end
