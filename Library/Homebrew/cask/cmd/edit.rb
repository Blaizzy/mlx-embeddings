# frozen_string_literal: true

module Cask
  class Cmd
    # Implementation of the `brew cask edit` command.
    #
    # @api private
    class Edit < AbstractCommand
      def self.min_named
        :cask
      end

      def self.max_named
        1
      end

      def self.description
        "Open the given <cask> for editing."
      end

      def initialize(*)
        super
      rescue Homebrew::CLI::MaxNamedArgumentsError
        raise UsageError, "Only one cask can be edited at a time."
      end

      def run
        exec_editor cask_path
      rescue CaskUnavailableError => e
        reason = e.reason.empty? ? +"" : +"#{e.reason} "
        reason.concat("Run #{Formatter.identifier("brew cask create #{e.token}")} to create a new Cask.")
        raise e.class.new(e.token, reason.freeze)
      end

      def cask_path
        casks.first.sourcefile_path
      rescue CaskInvalidError, CaskUnreadableError, MethodDeprecatedError
        path = CaskLoader.path(args.first)
        return path if path.file?

        raise
      end
    end
  end
end
