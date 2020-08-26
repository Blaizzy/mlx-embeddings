# frozen_string_literal: true

module Cask
  class Cmd
    # Implementation of the `brew cask _help` command.
    #
    # @api private
    class InternalHelp < AbstractInternalCommand
      def self.max_named
        0
      end

      def self.description
        "Print help for unstable internal-use commands."
      end

      def run
        max_command_len = Cmd.commands.map(&:length).max
        puts "Unstable Internal-use Commands:\n\n"
        Cmd.command_classes.each do |klass|
          next if klass.visible?

          puts "    #{klass.command_name.ljust(max_command_len)}  #{klass.help}"
        end
        puts "\n"
      end
    end
  end
end
