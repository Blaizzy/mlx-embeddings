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
      check_cpu
      attempt_directory_creation
      check_cc_argv(cc)
      diagnostic_checks(:supported_configuration_checks, fatal: all_fatal)
      diagnostic_checks(:fatal_preinstall_checks)
    end
    alias generic_perform_preinstall_checks perform_preinstall_checks
    module_function :generic_perform_preinstall_checks

    def perform_build_from_source_checks(all_fatal: false)
      diagnostic_checks(:fatal_build_from_source_checks)
      diagnostic_checks(:build_from_source_checks, fatal: all_fatal)
    end

    def check_cpu
      return if Hardware::CPU.intel? && Hardware::CPU.is_64_bit?

      message = "Sorry, Homebrew does not support your computer's CPU architecture!"
      if Hardware::CPU.arm?
        opoo message
        return
      elsif Hardware::CPU.ppc?
        message += <<~EOS
          For PowerPC Mac (PPC32/PPC64BE) support, see:
            #{Formatter.url("https://github.com/mistydemeo/tigerbrew")}
        EOS
      end
      abort message
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

    def diagnostic_checks(type, fatal: true)
      @checks ||= Diagnostic::Checks.new
      failed = false
      @checks.public_send(type).each do |check|
        out = @checks.public_send(check)
        next if out.nil?

        if fatal
          failed ||= true
          ofail out
        else
          opoo out
        end
      end
      exit 1 if failed && fatal
    end
    private_class_method :diagnostic_checks
  end
end

require "extend/os/install"
