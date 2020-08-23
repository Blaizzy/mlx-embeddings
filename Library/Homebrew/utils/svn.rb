# frozen_string_literal: true

require "system_command"

module Utils
  # Helper functions for querying SVN information.
  #
  # @api private
  module Svn
    module_function

    def available?
      version.present?
    end

    def version
      return @version if defined?(@version)

      stdout, _, status = system_command(HOMEBREW_SHIMS_PATH/"scm/svn", args: ["--version"], print_stderr: false)
      @version = status.success? ? stdout.chomp[/svn, version (\d+(?:\.\d+)*)/, 1] : nil
    end

    def remote_exists?(url)
      return true unless available?

      # OK to unconditionally trust here because we're just checking if
      # a URL exists.
      quiet_system "svn", "ls", url, "--depth", "empty",
                   "--non-interactive", "--trust-server-cert"
    end

    def clear_version_cache
      remove_instance_variable(:@version) if defined?(@version)
    end
  end
end
