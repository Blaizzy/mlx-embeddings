# frozen_string_literal: true

module Cask
  class Cmd
    # Implementation of the `brew cask help` command.
    #
    # @api private
    class Help < AbstractCommand
      def self.max_named
        1
      end

      def self.description
        "Print help for `cask` commands."
      end

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
        Cmd.command_classes.select(&:visible?).map { |klass| [klass.command_name, klass] }.to_h
      end
    end
  end
end
