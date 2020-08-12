# frozen_string_literal: true

module Utils
  def self.clear_svn_version_cache
    remove_instance_variable(:@svn_available) if defined?(@svn_available)
    remove_instance_variable(:@svn_version) if defined?(@svn_version)
  end

  def self.svn_available?
    return @svn_available if defined?(@svn_available)

    @svn_available = quiet_system HOMEBREW_SHIMS_PATH/"scm/svn", "--version"
  end

  def self.svn_version
    return unless svn_available?
    return @svn_version if defined?(@svn_version)

    @svn_version = Utils.popen_read(
      HOMEBREW_SHIMS_PATH/"scm/svn", "--version"
    ).chomp[/svn, version (\d+(?:\.\d+)*)/, 1]
  end

  def self.svn_remote_exists?(url)
    return true unless svn_available?

    # OK to unconditionally trust here because we're just checking if
    # a URL exists.
    quiet_system "svn", "ls", url, "--depth", "empty",
                 "--non-interactive", "--trust-server-cert"
  end
end
