# frozen_string_literal: true

require "optparse"
require "shellwords"

require "extend/optparse"

require "cask/config"

require "cask/cmd/options"

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

    include Options

    option "--appdir=PATH",               ->(value) { Config.global.appdir               = value }
    option "--colorpickerdir=PATH",       ->(value) { Config.global.colorpickerdir       = value }
    option "--prefpanedir=PATH",          ->(value) { Config.global.prefpanedir          = value }
    option "--qlplugindir=PATH",          ->(value) { Config.global.qlplugindir          = value }
    option "--mdimporterdir=PATH",        ->(value) { Config.global.mdimporterdir        = value }
    option "--dictionarydir=PATH",        ->(value) { Config.global.dictionarydir        = value }
    option "--fontdir=PATH",              ->(value) { Config.global.fontdir              = value }
    option "--servicedir=PATH",           ->(value) { Config.global.servicedir           = value }
    option "--input_methoddir=PATH",      ->(value) { Config.global.input_methoddir      = value }
    option "--internet_plugindir=PATH",   ->(value) { Config.global.internet_plugindir   = value }
    option "--audio_unit_plugindir=PATH", ->(value) { Config.global.audio_unit_plugindir = value }
    option "--vst_plugindir=PATH",        ->(value) { Config.global.vst_plugindir        = value }
    option "--vst3_plugindir=PATH",       ->(value) { Config.global.vst3_plugindir       = value }
    option "--screen_saverdir=PATH",      ->(value) { Config.global.screen_saverdir      = value }

    option "--help", :help, false

    option "--language=a,b,c", ->(value) { Config.global.languages = value }

    # override default handling of --version
    option "--version", ->(*) { raise OptionParser::InvalidOption }

    def self.command_classes
      @command_classes ||= constants.map(&method(:const_get))
                                    .select { |klass| klass.respond_to?(:run) }
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
      @args = process_options(*args)
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
      MacOS.full_version = ENV["MACOS_VERSION"] unless ENV["MACOS_VERSION"].nil?
      Tap.default_cask_tap.install unless Tap.default_cask_tap.installed?

      args = @args.dup
      command, args = detect_internal_command(*args) || detect_external_command(*args) || [NullCommand.new, args]

      if help?
        Help.new(command.command_name).run
      else
        command.run(*args)
      end
    rescue CaskError, MethodDeprecatedError, ArgumentError, OptionParser::InvalidOption => e
      onoe e.message
      $stderr.puts e.backtrace if debug?
      exit 1
    rescue StandardError, ScriptError, NoMemoryError => e
      onoe e.message
      $stderr.puts Utils.error_message_with_suggestions
      $stderr.puts e.backtrace
      exit 1
    end

    def self.nice_listing(cask_list)
      cask_taps = {}
      cask_list.each do |c|
        user, repo, token = c.split "/"
        repo.sub!(/^homebrew-/i, "")
        cask_taps[token] ||= []
        cask_taps[token].push "#{user}/#{repo}"
      end
      list = []
      cask_taps.each do |token, taps|
        if taps.length == 1
          list.push token
        else
          taps.each { |r| list.push [r, token].join "/" }
        end
      end
      list.sort
    end

    def process_options(*args)
      non_options = []

      if idx = args.index("--")
        non_options += args.drop(idx)
        args = args.first(idx)
      end

      exclude_regex = /^--#{Regexp.union(*Config::DEFAULT_DIRS.keys.map(&Regexp.public_method(:escape)))}=/
      cask_opts = Shellwords.shellsplit(ENV.fetch("HOMEBREW_CASK_OPTS", ""))
                            .reject { |arg| arg.match?(exclude_regex) }

      all_args = cask_opts + args

      i = 0
      remaining = []

      while i < all_args.count
        begin
          arg = all_args[i]

          remaining << arg unless process_arguments([arg]).empty?
        rescue OptionParser::MissingArgument
          raise if i + 1 >= all_args.count

          args = all_args[i..(i + 1)]
          process_arguments(args)
          i += 1
        rescue OptionParser::InvalidOption
          remaining << arg
        end

        i += 1
      end

      remaining + non_options
    end

    class ExternalRubyCommand
      def initialize(command, path)
        @command_name = command.to_s.capitalize.to_sym
        @path = path
      end

      def run(*args)
        require @path

        klass = begin
          Cmd.const_get(@command_name)
        rescue NameError
          return
        end

        klass.run(*args)
      end
    end

    class ExternalCommand
      def initialize(path)
        @path = path
      end

      def run(*)
        exec @path, *ARGV[1..]
      end
    end

    class NullCommand
      def run(*args)
        if args.empty?
          ofail "No subcommand given.\n"
        else
          ofail "Unknown subcommand: #{args.first}"
        end

        $stderr.puts
        $stderr.puts Help.usage
      end

      def help
        run
      end
    end
  end
end
