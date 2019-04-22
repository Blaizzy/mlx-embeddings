# frozen_string_literal: true

require "requirement"

class XcodeRequirement < Requirement
  fatal true

  satisfy(build_env: false) { xcode_installed_version }

  def initialize(tags = [])
    @version = tags.shift if tags.first.to_s.match?(/(\d\.)+\d/)
    super(tags)
  end

  def xcode_installed_version
    return false unless MacOS::Xcode.installed?
    return false unless xcode_swift_compatability?
    return true unless @version

    MacOS::Xcode.version >= @version
  end

  def message
    version = " #{@version}" if @version
    message = <<~EOS
      A full installation of Xcode.app#{version} is required to compile
      this software. Installing just the Command Line Tools is not sufficient.
    EOS
    unless xcode_swift_compatability?
      message += <<~EOS

        Xcode >=10.2 requires macOS >=10.14.4 to build many formulae.
      EOS
    end
    if @version && Version.new(MacOS::Xcode.latest_version) < Version.new(@version)
      message + <<~EOS

        Xcode#{version} cannot be installed on macOS #{MacOS.version}.
        You must upgrade your version of macOS.
      EOS
    else
      message + <<~EOS

        Xcode can be installed from the App Store.
      EOS
    end
  end

  def inspect
    "#<#{self.class.name}: #{tags.inspect} version=#{@version.inspect}>"
  end

  private

  # TODO: when 10.14.4 and 10.2 have been around for long enough remove this
  # method in favour of requiring 10.14.4 and 10.2.
  def xcode_swift_compatability?
    return true if MacOS::Xcode.version < "10.2"
    return true if MacOS.full_version >= "10.14.4"

    MacOS.full_version < "10.14"
  end
end
