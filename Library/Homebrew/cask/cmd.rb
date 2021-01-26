# typed: false
# frozen_string_literal: true

require "optparse"
require "shellwords"

require "cli/parser"
require "extend/optparse"

require "cask/config"

require "cask/cmd/abstract_command"
require "cask/cmd/--cache"
require "cask/cmd/audit"
require "cask/cmd/cat"
require "cask/cmd/create"
require "cask/cmd/doctor"
require "cask/cmd/edit"
require "cask/cmd/fetch"
require "cask/cmd/help"
require "cask/cmd/home"
require "cask/cmd/info"
require "cask/cmd/install"
require "cask/cmd/list"
require "cask/cmd/outdated"
require "cask/cmd/reinstall"
require "cask/cmd/style"
require "cask/cmd/uninstall"
require "cask/cmd/upgrade"
require "cask/cmd/zap"

require "cask/cmd/abstract_internal_command"
require "cask/cmd/internal_help"
require "cask/cmd/internal_stanza"

module Cask
  # Implementation of the `brew cask` command-line interface.
  #
  # @api private
  class Cmd
    extend T::Sig

    include Context

    ALIASES = {
      "ls"       => "list",
      "homepage" => "home",
      "instal"   => "install", # gem does the same
      "uninstal" => "uninstall",
      "rm"       => "uninstall",
      "remove"   => "uninstall",
      "abv"      => "info",
      "dr"       => "doctor",
    }.freeze

    DEPRECATED_COMMANDS = {
      Cmd::Cache     => "brew --cache [--cask]",
      Cmd::Audit     => "brew audit [--cask]",
      Cmd::Cat       => "brew cat [--cask]",
      Cmd::Create    => "brew create --cask --set-name <name> <url>",
      Cmd::Doctor    => "brew doctor --verbose",
      Cmd::Edit      => "brew edit [--cask]",
      Cmd::Fetch     => "brew fetch [--cask]",
      Cmd::Help      => "brew help",
      Cmd::Home      => "brew home",
      Cmd::Info      => "brew info [--cask]",
      Cmd::Install   => "brew install [--cask]",
      Cmd::List      => "brew list [--cask]",
      Cmd::Outdated  => "brew outdated [--cask]",
      Cmd::Reinstall => "brew reinstall [--cask]",
      Cmd::Style     => "brew style",
      Cmd::Uninstall => "brew uninstall [--cask]",
      Cmd::Upgrade   => "brew upgrade [--cask]",
      Cmd::Zap       => "brew uninstall --zap [--cask]",
    }.freeze

    def self.parser(&block)
      Homebrew::CLI::Parser.new do
        if block
          instance_eval(&block)
        else
          usage_banner <<~EOS
            `cask` <command> [<options>] [<cask>]

            Homebrew Cask provides a friendly CLI workflow for the administration of macOS applications distributed as binaries.

            See also: `man brew`
          EOS
        end

        cask_options
      end
    end

    def self.command_classes
      @command_classes ||= constants.map(&method(:const_get))
                                    .select { |klass| klass.is_a?(Class) && klass < AbstractCommand }
                                    .reject(&:abstract?)
                                    .sort_by(&:command_name)
    end

    def self.commands
      @commands ||= command_classes.map(&:command_name)
    end

    def self.lookup_command(command_name)
      @lookup ||= Hash[commands.zip(command_classes)]
      command_name = ALIASES.fetch(command_name, command_name)
      @lookup.fetch(command_name, nil)
    end

    def self.aliases
      ALIASES
    end

    def self.run(*args)
      new(*args).run
    end

    def initialize(*args)
      @argv = args
    end

    def find_external_command(command)
      @tap_cmd_directories ||= Tap.cmd_directories
      @path ||= PATH.new(@tap_cmd_directories, ENV["HOMEBREW_PATH"])

      external_ruby_cmd = @tap_cmd_directories.map { |d| d/"brewcask-#{command}.rb" }
                                              .find(&:file?)
      external_ruby_cmd ||= which("brewcask-#{command}.rb", @path)

      if external_ruby_cmd
        ExternalRubyCommand.new(command, external_ruby_cmd)
      elsif external_command = which("brewcask-#{command}", @path)
        ExternalCommand.new(external_command)
      end
    end

    def detect_internal_command(*args)
      args.each_with_index do |arg, i|
        if command = self.class.lookup_command(arg)
          args.delete_at(i)
          return [command, args]
        elsif !arg.start_with?("-")
          break
        end
      end

      nil
    end

    def detect_external_command(*args)
      args.each_with_index do |arg, i|
        if command = find_external_command(arg)
          args.delete_at(i)
          return [command, args]
        elsif !arg.start_with?("-")
          break
        end
      end

      nil
    end

    def run
      argv = @argv

      args = self.class.parser.parse(argv, ignore_invalid_options: true)

      Tap.install_default_cask_tap_if_necessary

      command, argv = detect_internal_command(*argv) ||
                      detect_external_command(*argv) ||
                      [args.remaining.empty? ? NullCommand : UnknownSubcommand.new(args.remaining.first), argv]

      if (replacement = DEPRECATED_COMMANDS[command])
        odisabled "`brew cask #{command.command_name}`", replacement
      end

      if args.help?
        puts command.help
      else
        command.run(*argv)
      end
    rescue CaskError, MethodDeprecatedError, ArgumentError => e
      onoe e.message
      $stderr.puts e.backtrace if args.debug?
      exit 1
    end

    # Wrapper class for running an external Ruby command.
    class ExternalRubyCommand
      def initialize(command, path)
        @command_name = command.to_s.capitalize.to_sym
        @path = path
      end

      def run(*args)
        command_class&.run(*args)
      end

      def help
        command_class&.help
      end

      private

      def command_class
        return @command_class if defined?(@command_class)

        require @path

        @command_class = begin
          Cmd.const_get(@command_name)
        rescue NameError
          nil
        end
      end
    end

    # Wrapper class for running an external command.
    class ExternalCommand
      def initialize(path)
        @path = path
      end

      def run(*argv)
        exec @path, *argv
      end

      def help
        exec @path, "--help"
      end
    end

    # Helper class for showing help for unknown subcommands.
    class UnknownSubcommand
      def initialize(command_name)
        @command_name = command_name
      end

      def run(*)
        raise UsageError, "Subcommand `#{@command_name}` does not exist."
      end

      def help
        run
      end
    end

    # Helper class for showing help when no subcommand is given.
    class NullCommand
      def self.run(*)
        raise UsageError, "No subcommand given."
      end

      def self.help
        Cmd.parser.generate_help_text
      end
    end
  end
end
