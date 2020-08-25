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

      min_named :formula
    end
  end

  def formula
    args = formula_args.parse

    args.named.to_formulae_paths.each(&method(:puts))
  end
end
