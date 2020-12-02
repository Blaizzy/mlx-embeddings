# typed: true
# frozen_string_literal: true

module Cask
  class Cmd
    # Implementation of the `brew cask help` command.
    #
    # @api private
    class Help < AbstractCommand
      extend T::Sig

      sig { override.returns(T.nilable(Integer)) }
      def self.max_named
        1
      end

      sig { returns(String) }
      def self.description
        "Print help for `cask` commands."
      end

      sig { void }
      def run
        if args.named.empty?
          puts Cmd.parser.generate_help_text
        else
          command_name = args.named.first

          unless command = self.class.commands[command_name]
            raise "No help information found for command '#{command_name}'."
          end

          puts command.help
        end
      end

      def self.commands
        Cmd.command_classes.select(&:visible?).index_by(&:command_name)
      end
    end
  end
end
