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

    [["Built-in commands", -> { Commands.internal_commands }],
     ["Built-in developer commands", -> { Commands.internal_developer_commands }],
     ["External commands", -> { Commands.external_commands }],
     ["Cask commands", -> { Commands.cask_internal_commands }],
     ["External cask commands", -> { Commands.cask_external_commands }]]
      .each_with_index do |title_and_proc, index|
      title, proc = title_and_proc
      cmds = proc.call
      if cmds.present?
        puts unless index.zero?
        ohai title, Formatter.columns(cmds)
      end
    end
  end
end
