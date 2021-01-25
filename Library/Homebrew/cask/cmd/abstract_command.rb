# typed: false
# frozen_string_literal: true

require "search"

module Cask
  class Cmd
    # Abstract superclass for all Cask implementations of commands.
    #
    # @api private
    class AbstractCommand
      extend T::Sig
      extend T::Helpers

      include Homebrew::Search

      OPTIONS = [
        [:switch, "--[no-]binaries", {
          description: "Disable/enable linking of helper executables (default: enabled).",
          env:         :cask_opts_binaries,
        }],
        [:switch, "--require-sha",  {
          description: "Require all casks to have a checksum.",
          env:         :cask_opts_require_sha,
        }],
        [:switch, "--[no-]quarantine", {
          description: "Disable/enable quarantining of downloads (default: enabled).",
          env:         :cask_opts_quarantine,
        }],
      ].freeze

      def self.parser(&block)
        Cmd.parser do
          instance_eval(&block) if block

          OPTIONS.each do |option|
            send(*option)
          end
        end
      end

      sig { returns(String) }
      def self.command_name
        @command_name ||= name.sub(/^.*:/, "").gsub(/(.)([A-Z])/, '\1_\2').downcase
      end

      sig { returns(T::Boolean) }
      def self.abstract?
        name.split("::").last.match?(/^Abstract[^a-z]/)
      end

      sig { returns(T::Boolean) }
      def self.visible?
        true
      end

      sig { returns(String) }
      def self.help
        parser.generate_help_text
      end

      sig { returns(String) }
      def self.short_description
        description[/\A[^.]*\./]
      end

      def self.run(*args)
        new(*args).run
      end

      attr_reader :args

      def initialize(*args)
        @args = self.class.parser.parse(args)
      end

      private

      def casks(alternative: -> { [] })
        return @casks if defined?(@casks)

        @casks = args.named.empty? ? alternative.call : args.named.to_casks
      rescue CaskUnavailableError => e
        reason = [e.reason, *suggestion_message(e.token)].join(" ")
        raise e.class.new(e.token, reason)
      end

      def suggestion_message(cask_token)
        matches = search_casks(cask_token)

        if matches.one?
          "Did you mean '#{matches.first}'?"
        elsif !matches.empty?
          "Did you mean one of these?\n#{Formatter.columns(matches.take(20))}"
        end
      end
    end
  end
end
