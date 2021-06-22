# typed: true
# frozen_string_literal: true

# Helper functions for querying operating system information.
#
# @api private
module OS
  extend T::Sig

  # Check if the operating system is macOS.
  #
  # @api public
  sig { returns(T::Boolean) }
  def self.mac?
    return false if ENV["HOMEBREW_TEST_GENERIC_OS"]

    RbConfig::CONFIG["host_os"].include? "darwin"
  end

  # Check if the operating system is Linux.
  #
  # @api public
  sig { returns(T::Boolean) }
  def self.linux?
    return false if ENV["HOMEBREW_TEST_GENERIC_OS"]

    RbConfig::CONFIG["host_os"].include? "linux"
  end

  # Get the kernel version.
  #
  # @api public
  sig { returns(Version) }
  def self.kernel_version
    @kernel_version ||= Version.new(Utils.safe_popen_read("uname", "-r").chomp)
  end

  ::OS_VERSION = ENV["HOMEBREW_OS_VERSION"]

  CI_GLIBC_VERSION = "2.23"
  CI_OS_VERSION = "Ubuntu 16.04"

  if OS.mac?
    require "os/mac"
    # Don't tell people to report issues on unsupported configurations.
    if !OS::Mac.version.prerelease? &&
       !OS::Mac.version.outdated_release? &&
       ARGV.none? { |v| v.start_with?("--cc=") } &&
       ENV["HOMEBREW_PREFIX"] == "/usr/local"
      ISSUES_URL = "https://docs.brew.sh/Troubleshooting"
    end
    PATH_OPEN = "/usr/bin/open"
  elsif OS.linux?
    require "os/linux"
    ISSUES_URL = "https://docs.brew.sh/Troubleshooting"
    PATH_OPEN = "xdg-open"
  end
end
