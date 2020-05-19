# frozen_string_literal: true

module Cask
  class Cmd
    class Help < AbstractCommand
      def run
        if args.empty?
          puts self.class.purpose
          puts
          puts self.class.usage
        elsif args.count == 1
          command_name = args.first

          if (command = self.class.commands[command_name]) && command.respond_to?(:usage)
            puts command.usage
            return
          end

          raise "No help information found for command '#{command_name}'."
        else
          raise ArgumentError, "#{self.class.command_name} only takes up to one argument."
        end
      end

      def self.purpose
        <<~EOS
          Homebrew Cask provides a friendly CLI workflow for the administration
          of macOS applications distributed as binaries.
        EOS
      end

      def self.commands
        Cmd.command_classes.select(&:visible?).map { |klass| [klass.command_name, klass] }.to_h
      end

      def self.usage
        max_command_len = Cmd.commands.map(&:length).max

        "Commands:\n" +
          Cmd.command_classes
             .select(&:visible?)
             .map { |klass| "    #{klass.command_name.ljust(max_command_len)}  #{klass.help}\n" }
             .join +
          %Q(\nSee also "man brew-cask")
      end

      def self.help
        "print help strings for commands"
      end
    end
  end
end
