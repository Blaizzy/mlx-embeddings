# frozen_string_literal: true

require "system_config"
require "diagnostic"

module Cask
  class Cmd
    class Doctor < AbstractCommand
      def initialize(*)
        super
        return if args.empty?

        raise ArgumentError, "#{self.class.command_name} does not take arguments."
      end

      def summary_header
        "Cask's Doctor Checkup"
      end

      def run
        success = true

        checks = Homebrew::Diagnostic::Checks.new true
        checks.cask_checks.each do |check|
          out = checks.send(check)

          if out.present?
            success = false
            puts out
          end
        end

        raise CaskError, "There are some problems with your setup." unless success
      end

      def self.help
        "checks for configuration issues"
      end
    end
  end
end
