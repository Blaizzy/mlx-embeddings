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
      Cmd::Cache     => "brew --cache --cask",
      Cmd::Doctor    => "brew doctor --verbose",
      Cmd::Home      => "brew home",
      Cmd::List      => "brew list --cask",
      Cmd::Outdated  => "brew outdated --cask",
      Cmd::Reinstall => "brew reinstall",
      Cmd::Upgrade   => "brew upgrade --cask",
    }.freeze

    def self.description
      max_command_length = Cmd.commands.map(&:length).max

      command_lines = Cmd.command_classes
                         .select(&:visible?)
                         .map do |klass|
        "  - #{"`#{klass.command_name}`".ljust(max_command_length + 2)}  #{klass.short_description}\n"
      end

      <<~EOS
        Homebrew Cask provides a friendly CLI workflow for the administration of macOS applications distributed as binaries.

        Commands:
        #{command_lines.join}

        See also: `man brew`
      EOS
    end

    def self.parser(&block)
      Homebrew::CLI::Parser.new do
        if block_given?
          instance_eval(&block)
        else
          usage_banner <<~EOS
            `cask` <command> [<options>] [<cask>]

            #{Cmd.description}
          EOS
        end

        flag "--appdir=",
             description: "Target location for Applications. " \
                          "Default: `#{Config::DEFAULT_DIRS[:appdir]}`"
        flag "--colorpickerdir=",
             description: "Target location for Color Pickers. " \
                          "Default: `#{Config::DEFAULT_DIRS[:colorpickerdir]}`"
        flag "--prefpanedir=",
             description: "Target location for Preference Panes. " \
                          "Default: `#{Config::DEFAULT_DIRS[:prefpanedir]}`"
        flag "--qlplugindir=",
             description: "Target location for QuickLook Plugins. " \
                          "Default: `#{Config::DEFAULT_DIRS[:qlplugindir]}`"
        flag "--mdimporterdir=",
             description: "Target location for Spotlight Plugins. " \
                          "Default: `#{Config::DEFAULT_DIRS[:mdimporterdir]}`"
        flag "--dictionarydir=",
             description: "Target location for Dictionaries. " \
                          "Default: `#{Config::DEFAULT_DIRS[:dictionarydir]}`"
        flag "--fontdir=",
             description: "Target location for Fonts. " \
                          "Default: `#{Config::DEFAULT_DIRS[:fontdir]}`"
        flag "--servicedir=",
             description: "Target location for Services. " \
                          "Default: `#{Config::DEFAULT_DIRS[:servicedir]}`"
        flag "--input_methoddir=",
             description: "Target location for Input Methods. " \
                          "Default: `#{Config::DEFAULT_DIRS[:input_methoddir]}`"
        flag "--internet_plugindir=",
             description: "Target location for Internet Plugins. " \
                          "Default: `#{Config::DEFAULT_DIRS[:internet_plugindir]}`"
        flag "--audio_unit_plugindir=",
             description: "Target location for Audio Unit Plugins. " \
                          "Default: `#{Config::DEFAULT_DIRS[:audio_unit_plugindir]}`"
        flag "--vst_plugindir=",
             description: "Target location for VST Plugins. " \
                          "Default: `#{Config::DEFAULT_DIRS[:vst_plugindir]}`"
        flag "--vst3_plugindir=",
             description: "Target location for VST3 Plugins. " \
                          "Default: `#{Config::DEFAULT_DIRS[:vst3_plugindir]}`"
        flag "--screen_saverdir=",
             description: "Target location for Screen Savers. " \
                          "Default: `#{Config::DEFAULT_DIRS[:screen_saverdir]}`"
        comma_array "--language",
                    description: "Set language of the Cask to install. The first matching " \
                                 "language is used, otherwise the default language on the Cask. " \
                                 "The default value is the `language of your system`"
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

      Config::DEFAULT_DIRS.each_key do |name|
        Config.global.public_send(:"#{name}=", args[name]) if args[name]
      end

      Config.global.languages = args.language if args.language

      Tap.default_cask_tap.install unless Tap.default_cask_tap.installed?

      command, argv = detect_internal_command(*argv) ||
                      detect_external_command(*argv) ||
                      [args.remaining.empty? ? NullCommand : UnknownSubcommand.new(args.remaining.first), argv]

      if (replacement = DEPRECATED_COMMANDS[command])
        odeprecated "brew cask #{command.command_name}", replacement
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
