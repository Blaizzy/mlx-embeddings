# typed: false
# frozen_string_literal: true

require "tempfile"
require "utils/shell"
require "hardware"
require "os/linux/glibc"
require "os/linux/kernel"

module Homebrew
  module Diagnostic
    class Checks
      def supported_configuration_checks
        %w[
          check_glibc_minimum_version
          check_kernel_minimum_version
          check_supported_architecture
        ].freeze
      end

      def check_tmpdir_sticky_bit
        message = generic_check_tmpdir_sticky_bit
        return if message.nil?

        message + <<~EOS
          If you don't have administrative privileges on this machine,
          create a directory and set the HOMEBREW_TEMP environment variable,
          for example:
            install -d -m 1755 ~/tmp
            #{Utils::Shell.set_variable_in_profile("HOMEBREW_TEMP", "~/tmp")}
        EOS
      end

      def check_tmpdir_executable
        f = Tempfile.new(%w[homebrew_check_tmpdir_executable .sh], HOMEBREW_TEMP)
        f.write "#!/bin/sh\n"
        f.chmod 0700
        f.close
        return if system f.path

        <<~EOS
          The directory #{HOMEBREW_TEMP} does not permit executing
          programs. It is likely mounted as "noexec". Please set HOMEBREW_TEMP
          in your #{shell_profile} to a different directory, for example:
            export HOMEBREW_TEMP=~/tmp
            echo 'export HOMEBREW_TEMP=~/tmp' >> #{shell_profile}
        EOS
      ensure
        f.unlink
      end

      def check_xdg_data_dirs
        return if ENV["XDG_DATA_DIRS"].blank?
        return if ENV["XDG_DATA_DIRS"].split("/").include?(HOMEBREW_PREFIX/"share")

        <<~EOS
          Homebrew's share was not found in your XDG_DATA_DIRS but you have
          this variable set to include other locations.
          Some programs like `vapigen` may not work correctly.
          Consider adding Homebrew's share directory to XDG_DATA_DIRS like so:
            echo 'export XDG_DATA_DIRS="#{HOMEBREW_PREFIX}/share:$XDG_DATA_DIRS"' >> #{shell_profile}
        EOS
      end

      def check_umask_not_zero
        return unless File.umask.zero?

        <<~EOS
          umask is currently set to 000. Directories created by Homebrew cannot
          be world-writable. This issue can be resolved by adding "umask 002" to
          your #{shell_profile}:
            echo 'umask 002' >> #{shell_profile}
        EOS
      end

      def check_supported_architecture
        return if Hardware::CPU.arch == :x86_64

        <<~EOS
          Your CPU architecture (#{Hardware::CPU.arch}) is not supported. We only support
          x86_64 CPU architectures. You will be unable to use binary packages (bottles).
          #{please_create_pull_requests}
        EOS
      end

      def check_glibc_minimum_version
        return unless OS::Linux::Glibc.below_minimum_version?

        <<~EOS
          Your system glibc #{OS::Linux::Glibc.system_version} is too old.
          We only support glibc #{OS::Linux::Glibc.minimum_version} or later.
          #{please_create_pull_requests}
          We recommend updating to a newer version via your distribution's
          package manager, upgrading your distribution to the latest version,
          or changing distributions.
        EOS
      end

      def check_kernel_minimum_version
        return unless OS::Linux::Kernel.below_minimum_version?

        <<~EOS
          Your Linux kernel #{OS.kernel_version} is too old.
          We only support kernel #{OS::Linux::Kernel.minimum_version} or later.
          You will be unable to use binary packages (bottles).
          #{please_create_pull_requests}
          We recommend updating to a newer version via your distribution's
          package manager, upgrading your distribution to the latest version,
          or changing distributions.
        EOS
      end
    end
  end
end
