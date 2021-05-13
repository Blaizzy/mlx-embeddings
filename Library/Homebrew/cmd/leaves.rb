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
      switch "-r", "--installed-on-request",
             description: "Only list leaves that were manually installed."
      switch "-p", "--installed-as-dependency",
             description: "Only list leaves that were installed as dependencies."

      conflicts "--installed-on-request", "--installed-as-dependency"

      named_args :none
    end
  end

  def installed_on_request?(formula)
    Tab.for_keg(formula.any_installed_keg).installed_on_request
  end

  def installed_as_dependency?(formula)
    Tab.for_keg(formula.any_installed_keg).installed_as_dependency
  end

  def leaves
    args = leaves_args.parse

    leaves_list = Formula.installed_formulae_with_no_dependents

    leaves_list.select!(&method(:installed_on_request?)) if args.installed_on_request?
    leaves_list.select!(&method(:installed_as_dependency?)) if args.installed_as_dependency?

    leaves_list.map(&:full_name)
               .sort
               .each(&method(:puts))
  end
end
