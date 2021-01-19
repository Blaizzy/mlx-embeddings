# typed: true
# frozen_string_literal: true

require "formula"
require "cli/parser"

module Homebrew
  extend T::Sig

  module_function

  sig { returns(CLI::Parser) }
  def formula_args
    Homebrew::CLI::Parser.new do
      description <<~EOS
        Display the path where <formula> is located.
      EOS

      named_args :formula, min: 1
    end
  end

  def formula
    args = formula_args.parse

    args.named.to_formulae_paths.each(&method(:puts))
  end
end
