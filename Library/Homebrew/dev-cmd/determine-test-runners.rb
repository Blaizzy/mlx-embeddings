# typed: strict
# frozen_string_literal: true

require "cli/parser"
require "test_runner_formula"
require "github_runner_matrix"

module Homebrew
  extend T::Sig

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

  sig {
    params(
      version:          String,
      arch:             Symbol,
      ephemeral:        T::Boolean,
      ephemeral_suffix: T.nilable(String),
    ).returns(T::Hash[Symbol, T.any(String, T::Boolean)])
  }
  def self.runner_spec(version, arch:, ephemeral:, ephemeral_suffix: nil)
    case arch
    when :arm64 then { runner: "#{version}-arm64#{ephemeral_suffix}", clean: !ephemeral }
    when :x86_64 then { runner: "#{version}#{ephemeral_suffix}", clean: !ephemeral }
    else raise "Unexpected arch: #{arch}"
    end
  end

  sig { void }
  def self.determine_test_runners
    args = determine_test_runners_args.parse

    eval_all = args.eval_all? || Homebrew::EnvConfig.eval_all?

    odie "`--dependents` requires `--eval-all` or `HOMEBREW_EVAL_ALL`!" if args.dependents? && !eval_all

    Formulary.enable_factory_cache!

    testing_formulae = args.named.first.split(",")
    testing_formulae.map! { |name| TestRunnerFormula.new(Formulary.factory(name), eval_all: eval_all) }
                    .freeze
    deleted_formulae = args.named.second&.split(",")

    linux_runner       = ENV.fetch("HOMEBREW_LINUX_RUNNER") { raise "HOMEBREW_LINUX_RUNNER is not defined" }
    linux_cleanup      = ENV.fetch("HOMEBREW_LINUX_CLEANUP") { raise "HOMEBREW_LINUX_CLEANUP is not defined" }
    github_run_id      = ENV.fetch("GITHUB_RUN_ID") { raise "GITHUB_RUN_ID is not defined" }
    github_run_attempt = ENV.fetch("GITHUB_RUN_ATTEMPT") { raise "GITHUB_RUN_ATTEMPT is not defined" }
    github_output      = ENV.fetch("GITHUB_OUTPUT") { raise "GITHUB_OUTPUT is not defined" }

    linux_runner_spec = {
      runner:    linux_runner,
      container: {
        image:   "ghcr.io/homebrew/ubuntu22.04:master",
        options: "--user=linuxbrew -e GITHUB_ACTIONS_HOMEBREW_SELF_HOSTED",
      },
      workdir:   "/github/home",
      timeout:   4320,
      cleanup:   linux_cleanup == "true",
    }
    ephemeral_suffix = "-#{github_run_id}-#{github_run_attempt}"

    available_runners = []
    available_runners << { platform: :linux, arch: :x86_64, runner_spec: linux_runner_spec, macos_version: nil }

    MacOSVersions::SYMBOLS.each_value do |version|
      macos_version = OS::Mac::Version.new(version)
      next if macos_version.outdated_release? || macos_version.prerelease?

      spec = runner_spec(version, arch: :x86_64, ephemeral: true, ephemeral_suffix: ephemeral_suffix)
      available_runners << { platform: :macos, arch: :x86_64, runner_spec: spec, macos_version: macos_version }

      # Use bare metal runner when testing dependents on ARM64 Monterey.
      if (macos_version >= :ventura && args.dependents?) || macos_version >= :monterey
        spec = runner_spec(version, arch: :arm64, ephemeral: true, ephemeral_suffix: ephemeral_suffix)
        available_runners << { platform: :macos, arch: :arm64, runner_spec: spec, macos_version: macos_version }
      elsif macos_version >= :big_sur
        spec = runner_spec(version, arch: :arm64, ephemeral: false)
        available_runners << { platform: :macos, arch: :arm64, runner_spec: spec, macos_version: macos_version }
      end
    end

    runner_matrix = GitHubRunnerMatrix.new(
      available_runners,
      testing_formulae,
      deleted_formulae,
      dependent_matrix: args.dependents?,
    )
    runners = runner_matrix.active_runners

    if !args.dependents? && runners.blank?
      # If there are no tests to run, add a runner that is meant to do nothing
      # to support making the `tests` job a required status check.
      runners << { runner: "ubuntu-latest", no_op: true }
    end

    File.open(github_output, "a") do |f|
      f.puts("runners=#{runners.to_json}")
      f.puts("runners_present=#{runners.present?}")
    end
  end
end
