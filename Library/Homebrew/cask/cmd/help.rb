# frozen_string_literal: true

module Cask
  class Cmd
    class Help < AbstractCommand
      def initialize(*)
        super
        return if args.empty?

        raise ArgumentError, "#{self.class.command_name} does not take arguments."
      end

      def run
        puts self.class.purpose
        puts
        puts self.class.usage
      end

      def self.purpose
        <<~EOS
          Homebrew Cask provides a friendly CLI workflow for the administration
          of macOS applications distributed as binaries.
        EOS
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
