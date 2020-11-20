# typed: false
# frozen_string_literal: true

require "json"
require "style"

module Cask
  class Cmd
    # Implementation of the `brew cask style` command.
    #
    # @api private
    class Style < AbstractCommand
      extend T::Sig

      sig { returns(String) }
      def self.description
        "Checks style of the given <cask> using RuboCop."
      end

      def self.parser
        super do
          switch "--fix",
                 description: "Fix style violations automatically using RuboCop's auto-correct feature."
        end
      end

      sig { void }
      def run
        success = Homebrew::Style.check_style_and_print(
          cask_paths,
          fix:     args.fix?,
          debug:   args.debug?,
          verbose: args.verbose?,
        )
        raise CaskError, "Style check failed." unless success
      end

      def cask_paths
        @cask_paths ||= if args.named.empty?
          Tap.map(&:cask_dir).select(&:directory?)
        elsif args.named.any? { |file| File.exist?(file) }
          args.named.map { |path| Pathname(path).expand_path }
        else
          casks.map(&:sourcefile_path)
        end
      end
    end
  end
end
