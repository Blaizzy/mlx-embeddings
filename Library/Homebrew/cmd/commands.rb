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
      puts Formatter.columns(Commands.commands(aliases: args.include_aliases?))
      return
    end

    ohai "Built-in commands", Formatter.columns(Commands.internal_commands)
    puts
    ohai "Built-in developer commands", Formatter.columns(Commands.internal_developer_commands)

    external_commands = Commands.external_commands
    return if external_commands.blank?

    puts
    ohai "External commands", Formatter.columns(external_commands)
  end
end
