# typed: false
# frozen_string_literal: true

require "cli/parser"

module Homebrew
  extend T::Sig

  module_function

  sig { returns(CLI::Parser) }
  def commands_args
    Homebrew::CLI::Parser.new do
      description <<~EOS
        Show lists of built-in and external commands.
      EOS
      switch "-q", "--quiet",
             description: "List only the names of commands without category headers."
      switch "--include-aliases",
             depends_on:  "--quiet",
             description: "Include aliases of internal commands."

      named_args :none
    end
  end

  def commands
    args = commands_args.parse

    if args.quiet?
      puts Formatter.columns(Commands.commands(aliases: args.include_aliases?))
      return
    end

    prepend_separator = false

    {
      "Built-in commands"           => Commands.internal_commands,
      "Built-in developer commands" => Commands.internal_developer_commands,
      "External commands"           => Commands.external_commands,
      "Cask commands"               => Commands.cask_internal_commands,
      "External cask commands"      => Commands.cask_external_commands,
    }.each do |title, commands|
      next if commands.blank?

      puts if prepend_separator
      ohai title, Formatter.columns(commands)

      prepend_separator ||= true
    end
  end
end
