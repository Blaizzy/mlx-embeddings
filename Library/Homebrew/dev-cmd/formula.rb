# frozen_string_literal: true

require "formula"
require "cli/parser"

module Homebrew
  module_function

  def formula_args
    Homebrew::CLI::Parser.new do
      usage_banner <<~EOS
        `formula` <formula>

        Display the path where <formula> is located.
      EOS
      switch :verbose
      switch :debug
    end
  end

  def formula
    formula_args.parse

    raise FormulaUnspecifiedError if Homebrew.args.named.blank?

    Homebrew.args.resolved_formulae.each { |f| puts f.path }
  end
end
