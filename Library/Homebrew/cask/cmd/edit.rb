# typed: true
# frozen_string_literal: true

module Cask
  class Cmd
    # Implementation of the `brew cask edit` command.
    #
    # @api private
    class Edit < AbstractCommand
      extend T::Sig

      sig { override.returns(T.nilable(T.any(Integer, Symbol))) }
      def self.min_named
        :cask
      end

      sig { override.returns(T.nilable(Integer)) }
      def self.max_named
        1
      end

      sig { returns(String) }
      def self.description
        "Open the given <cask> for editing."
      end

      def initialize(*)
        super
      rescue Homebrew::CLI::MaxNamedArgumentsError
        raise UsageError, "Only one cask can be edited at a time."
      end

      sig { void }
      def run
        exec_editor cask_path
      rescue CaskUnavailableError => e
        reason = e.reason.empty? ? +"" : +"#{e.reason} "
        reason.concat(
          "Run #{Formatter.identifier("brew create --cask --set-name #{e.token} <url>")} to create a new Cask.",
        )
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
