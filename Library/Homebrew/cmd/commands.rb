# frozen_string_literal: true

require "cli/parser"

module Homebrew
  module_function

  def commands_args
    Homebrew::CLI::Parser.new do
      usage_banner <<~EOS
        `commands` [<options>]

        Show lists of built-in and external commands.
      EOS
      switch :quiet,
             description: "List only the names of commands without category headers."
      switch "--include-aliases",
             depends_on:  "--quiet",
             description: "Include aliases of internal commands."
      switch :verbose
      switch :debug
      max_named 0
    end
  end

  def commands
    commands_args.parse

    if args.quiet?
      cmds = internal_commands
      cmds += external_commands
      cmds += internal_developer_commands
      cmds += HOMEBREW_INTERNAL_COMMAND_ALIASES.keys if args.include_aliases?
      puts Formatter.columns(cmds.sort)
      return
    end

    # Find commands in Homebrew/cmd
    ohai "Built-in commands", Formatter.columns(internal_commands.sort)

    # Find commands in Homebrew/dev-cmd
    puts
    ohai "Built-in developer commands", Formatter.columns(internal_developer_commands.sort)

    exts = external_commands
    return if exts.empty?

    # Find commands in the PATH
    puts
    ohai "External commands", Formatter.columns(exts)
  end

  def internal_commands
    find_internal_commands HOMEBREW_LIBRARY_PATH/"cmd"
  end

  def internal_developer_commands
    find_internal_commands HOMEBREW_LIBRARY_PATH/"dev-cmd"
  end

  def external_commands
    cmd_paths = PATH.new(ENV["PATH"]).append(Tap.cmd_directories)
    cmd_paths.each_with_object([]) do |path, cmds|
      Dir["#{path}/brew-*"].each do |file|
        next unless File.executable?(file)

        cmd = File.basename(file, ".rb")[5..-1]
        next if cmd.include?(".")

        cmds << cmd
      end
    end.sort
  end

  def find_internal_commands(directory)
    Pathname.glob(directory/"*")
            .select(&:file?)
            .map { |f| f.basename.to_s.sub(/\.(?:rb|sh)$/, "") }
  end
end
