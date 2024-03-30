# typed: strong
# frozen_string_literal: true

require "abstract_command"
require "help"

module Homebrew
  module Cmd
    class HelpCmd < AbstractCommand
      cmd_args { description "Print help information." }

      sig { override.void }
      def run
        Help.help
      end
    end
  end
end
