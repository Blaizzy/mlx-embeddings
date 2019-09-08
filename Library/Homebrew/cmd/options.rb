# frozen_string_literal: true

require "formula"
require "options"
require "cli/parser"

module Homebrew
  module_function

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
      switch :debug
      conflicts "--installed", "--all"
    end
  end

  def options
    options_args.parse

    if args.all?
      puts_options Formula.to_a.sort
    elsif args.installed?
      puts_options Formula.installed.sort
    else
      raise FormulaUnspecifiedError if args.remaining.empty?

      puts_options Homebrew.args.formulae
    end
  end

  def puts_options(formulae)
    formulae.each do |f|
      next if f.options.empty?

      if args.compact?
        puts f.options.as_flags.sort * " "
      else
        puts f.full_name if formulae.length > 1
        dump_options_for_formula f
        puts
      end
    end
  end
end
