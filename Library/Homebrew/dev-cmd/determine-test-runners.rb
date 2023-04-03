# typed: strict
# frozen_string_literal: true

require "cli/parser"

module Homebrew
  extend T::Sig

  sig { returns(Homebrew::CLI::Parser) }
  def self.determine_test_runners_args
    Homebrew::CLI::Parser.new do
      usage_banner <<~EOS
        `determine-test-runners` <testing-formulae> [<deleted-formulae>]

        Determines the runners used to test formulae or their dependents.
      EOS
      switch "--dependents",
             description: "Determine runners for testing dependents."

      named_args min: 1, max: 2

      hide_from_man_page!
    end
  end

  sig { void }
  def self.determine_test_runners
    odie "This command is supported only on Linux!"
  end
end

require "extend/os/dev-cmd/determine-test-runners"
