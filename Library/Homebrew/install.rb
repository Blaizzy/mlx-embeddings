# typed: true
# frozen_string_literal: true

require "diagnostic"
require "fileutils"
require "hardware"
require "development_tools"

module Homebrew
  # Helper module for performing (pre-)install checks.
  #
  # @api private
  module Install
    module_function

    def perform_preinstall_checks(all_fatal: false, cc: nil)
      check_prefix
      check_cpu
      attempt_directory_creation
      check_cc_argv(cc)
      Diagnostic.checks(:supported_configuration_checks, fatal: all_fatal)
      Diagnostic.checks(:fatal_preinstall_checks)
    end
    alias generic_perform_preinstall_checks perform_preinstall_checks
    module_function :generic_perform_preinstall_checks

    def perform_build_from_source_checks(all_fatal: false)
      Diagnostic.checks(:fatal_build_from_source_checks)
      Diagnostic.checks(:build_from_source_checks, fatal: all_fatal)
    end

    def check_prefix
      if (Hardware::CPU.intel? || Hardware::CPU.in_rosetta2?) &&
         HOMEBREW_PREFIX.to_s == HOMEBREW_MACOS_ARM_DEFAULT_PREFIX
        if Hardware::CPU.in_rosetta2?
          odie <<~EOS
            Cannot install under Rosetta 2 in ARM default prefix (#{HOMEBREW_PREFIX})!
            To rerun under ARM use:
                arch -arm64 brew install ...
            To install under x86_64, install Homebrew into #{HOMEBREW_DEFAULT_PREFIX}.
          EOS
        else
          odie "Cannot install on Intel processor in ARM default prefix (#{HOMEBREW_PREFIX})!"
        end
      elsif Hardware::CPU.arm? && HOMEBREW_PREFIX.to_s == HOMEBREW_DEFAULT_PREFIX
        odie <<~EOS
          Cannot install in Homebrew on ARM processor in Intel default prefix (#{HOMEBREW_PREFIX})!
          Please create a new installation in #{HOMEBREW_MACOS_ARM_DEFAULT_PREFIX} using one of the
          "Alternative Installs" from:
            #{Formatter.url("https://docs.brew.sh/Installation")}
          You can migrate your previously installed formula list with:
            brew bundle dump
        EOS
      end
    end

    def check_cpu
      return unless Hardware::CPU.ppc?

      odie <<~EOS
        Sorry, Homebrew does not support your computer's CPU architecture!
        For PowerPC Mac (PPC32/PPC64BE) support, see:
          #{Formatter.url("https://github.com/mistydemeo/tigerbrew")}
      EOS
    end
    private_class_method :check_cpu

    def attempt_directory_creation
      Keg::MUST_EXIST_DIRECTORIES.each do |dir|
        FileUtils.mkdir_p(dir) unless dir.exist?

        # Create these files to ensure that these directories aren't removed
        # by the Catalina installer.
        # (https://github.com/Homebrew/brew/issues/6263)
        keep_file = dir/".keepme"
        FileUtils.touch(keep_file) unless keep_file.exist?
      rescue
        nil
      end
    end
    private_class_method :attempt_directory_creation

    def check_cc_argv(cc)
      return unless cc

      @checks ||= Diagnostic::Checks.new
      opoo <<~EOS
        You passed `--cc=#{cc}`.
        #{@checks.please_create_pull_requests}
      EOS
    end
    private_class_method :check_cc_argv
  end
end

require "extend/os/install"
