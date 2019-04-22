# frozen_string_literal: true

module OS
  def self.mac?
    return false if ENV["HOMEBREW_TEST_GENERIC_OS"]

    RbConfig::CONFIG["host_os"].include? "darwin"
  end

  def self.linux?
    return false if ENV["HOMEBREW_TEST_GENERIC_OS"]

    RbConfig::CONFIG["host_os"].include? "linux"
  end

  ::OS_VERSION = ENV["HOMEBREW_OS_VERSION"]

  if OS.mac?
    require "os/mac"
    # Don't tell people to report issues on unsupported configurations.
    if !OS::Mac.prerelease? &&
       !OS::Mac.outdated_release? &&
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
