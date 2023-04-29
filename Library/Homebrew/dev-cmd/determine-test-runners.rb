# typed: strict
# frozen_string_literal: true

require "cli/parser"
require "test_runner_formula"
require "github_runner_matrix"

module Homebrew
  sig { returns(Homebrew::CLI::Parser) }
  def self.determine_test_runners_args
    Homebrew::CLI::Parser.new do
      usage_banner <<~EOS
        `determine-test-runners` <testing-formulae> [<deleted-formulae>]

        Determines the runners used to test formulae or their dependents.
      EOS
      switch "--eval-all",
             description: "Evaluate all available formulae, whether installed or not, to determine testing " \
                          "dependents."
      switch "--dependents",
             description: "Determine runners for testing dependents. Requires `--eval-all` or `HOMEBREW_EVAL_ALL`."

      named_args min: 1, max: 2

      hide_from_man_page!
    end
  end

  sig { void }
  def self.determine_test_runners
    args = determine_test_runners_args.parse

    eval_all = args.eval_all? || Homebrew::EnvConfig.eval_all?

    odie "`--dependents` requires `--eval-all` or `HOMEBREW_EVAL_ALL`!" if args.dependents? && !eval_all

    testing_formulae = args.named.first.split(",")
    testing_formulae.map! { |name| TestRunnerFormula.new(Formulary.factory(name), eval_all: eval_all) }
                    .freeze
    deleted_formulae = args.named.second&.split(",").freeze

    runner_matrix = GitHubRunnerMatrix.new(testing_formulae, deleted_formulae, dependent_matrix: args.dependents?)
    runners = runner_matrix.active_runner_specs_hash

    ohai "Runners", JSON.pretty_generate(runners)

    github_output = ENV.fetch("GITHUB_OUTPUT")
    File.open(github_output, "a") do |f|
      f.puts("runners=#{runners.to_json}")
      f.puts("runners_present=#{runners.present?}")
    end
  end
end
