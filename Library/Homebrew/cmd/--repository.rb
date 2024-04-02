# typed: strict
# frozen_string_literal: true

require "abstract_command"

module Homebrew
  module Cmd
    class Repository < AbstractCommand
      sig { override.returns(String) }
      def self.command_name = "--repository"

      cmd_args do
        description <<~EOS
          Display where Homebrew's Git repository is located.

          If <user>`/`<repo> are provided, display where tap <user>`/`<repo>'s directory is located.
        EOS

        named_args :tap
      end

      sig { override.void }
      def run
        if args.no_named?
          puts HOMEBREW_REPOSITORY
        else
          puts args.named.to_taps.map(&:path)
        end
      end
    end
  end
end
