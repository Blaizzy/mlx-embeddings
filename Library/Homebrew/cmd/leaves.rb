# typed: true
# frozen_string_literal: true

require "formula"
require "cli/parser"

module Homebrew
  extend T::Sig

  module_function

  sig { returns(CLI::Parser) }
  def leaves_args
    Homebrew::CLI::Parser.new do
      description <<~EOS
        List installed formulae that are not dependencies of another installed formula.
      EOS

      named_args :none
    end
  end

  def leaves
    leaves_args.parse

    Formula.installed_formulae_with_no_dependents.map(&:full_name).sort.each(&method(:puts))
  end
end
