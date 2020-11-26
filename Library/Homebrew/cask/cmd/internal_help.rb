# typed: strict
# frozen_string_literal: true

module Cask
  class Cmd
    # Implementation of the `brew cask _help` command.
    #
    # @api private
    class InternalHelp < AbstractInternalCommand
      extend T::Sig

      sig { override.returns(T.nilable(Integer)) }
      def self.max_named
        0
      end

      sig { returns(String) }
      def self.description
        "Print help for unstable internal-use commands."
      end

      sig { void }
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
