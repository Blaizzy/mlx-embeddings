# typed: false
# frozen_string_literal: true

require "formula"
require "options"
require "cli/parser"
require "commands"

module Homebrew
  extend T::Sig

  module_function

  sig { returns(CLI::Parser) }
  def options_args
    Homebrew::CLI::Parser.new do
      usage_banner <<~EOS
        `options` [<options>] [<formula>]

        Show install options specific to <formula>.
      EOS
      switch "--compact",
             description: "Show all options on a single line separated by spaces."
      switch "--installed",
             description: "Show options for formulae that are currently installed."
      switch "--all",
             description: "Show options for all available formulae."
      flag   "--command=",
             description: "Show options for the specified <command>."

      conflicts "--installed", "--all", "--command"

      named_args :formula
    end
  end

  def options
    args = options_args.parse

    if args.all?
      puts_options Formula.to_a.sort, args: args
    elsif args.installed?
      puts_options Formula.installed.sort, args: args
    elsif !args.command.nil?
      path = Commands.path(args.command)
      odie "Unknown command: #{args.command}" unless path
      cmd_options = if cmd_parser = CLI::Parser.from_cmd_path(path)
        cmd_parser.processed_options.map do |short, long, _, desc|
          [long || short, desc]
        end
      else
        cmd_comment_options(path)
      end
      if args.compact?
        puts cmd_options.sort.map(&:first) * " "
      else
        cmd_options.sort.each { |option, desc| puts "#{option}\n\t#{desc}" }
        puts
      end
    elsif args.no_named?
      raise FormulaUnspecifiedError
    else
      puts_options args.named.to_formulae, args: args
    end
  end

  def cmd_comment_options(cmd_path)
    options = []
    comment_lines = cmd_path.read.lines.grep(/^#:/)
    return options if comment_lines.empty?

    # skip the comment's initial usage summary lines
    comment_lines.slice(2..-1).each do |line|
      if / (?<option>-[-\w]+) +(?<desc>.*)$/ =~ line
        options << [option, desc]
      end
    end
    options
  end

  def puts_options(formulae, args:)
    formulae.each do |f|
      next if f.options.empty?

      if args.compact?
        puts f.options.as_flags.sort * " "
      else
        puts f.full_name if formulae.length > 1
        Options.dump_for_formula f
        puts
      end
    end
  end
end
