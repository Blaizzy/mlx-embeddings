# typed: strict
# frozen_string_literal: true

require "cli/parser"
require "formula"

class TestRunnerFormula
  extend T::Sig

  sig { returns(String) }
  attr_reader :name

  sig { returns(Formula) }
  attr_reader :formula

  sig { returns(T::Boolean) }
  attr_reader :eval_all

  sig { params(formula: Formula, eval_all: T::Boolean).void }
  def initialize(formula, eval_all: Homebrew::EnvConfig.eval_all?)
    @formula = T.let(formula, Formula)
    @name = T.let(formula.name, String)
    @dependent_hash = T.let({}, T::Hash[Symbol, T::Array[TestRunnerFormula]])
    @eval_all = T.let(eval_all, T::Boolean)
    freeze
  end

  sig { returns(T::Boolean) }
  def macos_only?
    formula.requirements.any? { |r| r.is_a?(MacOSRequirement) && !r.version_specified? }
  end

  sig { returns(T::Boolean) }
  def macos_compatible?
    !linux_only?
  end

  sig { returns(T::Boolean) }
  def linux_only?
    formula.requirements.any?(LinuxRequirement)
  end

  sig { returns(T::Boolean) }
  def linux_compatible?
    !macos_only?
  end

  sig { returns(T::Boolean) }
  def x86_64_only?
    formula.requirements.any? { |r| r.is_a?(ArchRequirement) && (r.arch == :x86_64) }
  end

  sig { returns(T::Boolean) }
  def x86_64_compatible?
    !arm64_only?
  end

  sig { returns(T::Boolean) }
  def arm64_only?
    formula.requirements.any? { |r| r.is_a?(ArchRequirement) && (r.arch == :arm64) }
  end

  sig { returns(T::Boolean) }
  def arm64_compatible?
    !x86_64_only?
  end

  sig { returns(T.nilable(MacOSRequirement)) }
  def versioned_macos_requirement
    formula.requirements.find { |r| r.is_a?(MacOSRequirement) && r.version_specified? }
  end

  sig { params(macos_version: OS::Mac::Version).returns(T::Boolean) }
  def compatible_with?(macos_version)
    # Assign to a variable to assist type-checking.
    requirement = versioned_macos_requirement
    return true if requirement.blank?

    macos_version.public_send(requirement.comparator, requirement.version)
  end

  SIMULATE_SYSTEM_SYMBOLS = T.let({ arm64: :arm, x86_64: :intel }.freeze, T::Hash[Symbol, Symbol])

  sig {
    params(
      platform:      Symbol,
      arch:          Symbol,
      macos_version: T.nilable(Symbol),
    ).returns(T::Array[TestRunnerFormula])
  }
  def dependents(platform:, arch:, macos_version:)
    cache_key = :"#{platform}_#{arch}_#{macos_version}"

    @dependent_hash.fetch(cache_key) do
      all = eval_all || Homebrew::EnvConfig.eval_all?
      formula_selector, eval_all_env = if all
        [:all, "1"]
      else
        [:installed, nil]
      end

      with_env(HOMEBREW_EVAL_ALL: eval_all_env) do
        Formulary.clear_cache
        Homebrew::SimulateSystem.arch = SIMULATE_SYSTEM_SYMBOLS.fetch(arch)
        Homebrew::SimulateSystem.os = macos_version || platform

        Formula.send(formula_selector)
               .select { |candidate_f| candidate_f.deps.map(&:name).include?(name) }
               .map { |f| TestRunnerFormula.new(f, eval_all: all) }
               .freeze
      ensure
        Homebrew::SimulateSystem.clear
      end
    end
  end
end

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
      testing_formulae: T::Array[TestRunnerFormula],
      platform:         Symbol,
      arch:             Symbol,
      macos_version:    T.nilable(OS::Mac::Version),
    ).returns(T::Boolean)
  }
  def self.formulae_have_untested_dependents?(testing_formulae, platform:, arch:, macos_version:)
    testing_formulae.any? do |formula|
      # If the formula has a platform/arch/macOS version requirement, then its
      # dependents don't need to be tested if these requirements are not satisfied.
      next false unless formula.send(:"#{platform}_compatible?")
      next false unless formula.send(:"#{arch}_compatible?")
      next false if macos_version && !formula.compatible_with?(macos_version)

      compatible_dependents = formula.dependents(platform: platform, arch: arch, macos_version: macos_version&.to_sym)
                                     .dup

      compatible_dependents.select! { |dependent_f| dependent_f.send(:"#{platform}_compatible?") }
      compatible_dependents.select! { |dependent_f| dependent_f.send(:"#{arch}_compatible?") }
      compatible_dependents.select! { |dependent_f| dependent_f.compatible_with?(macos_version) } if macos_version

      (compatible_dependents - testing_formulae).present?
    end
  end

  sig {
    params(
      formulae:         T::Array[TestRunnerFormula],
      dependents:       T::Boolean,
      deleted_formulae: T.nilable(T::Array[String]),
      platform:         Symbol,
      arch:             Symbol,
      macos_version:    T.nilable(OS::Mac::Version),
    ).returns(T::Boolean)
  }
  def self.add_runner?(formulae, dependents:, deleted_formulae:, platform:, arch:, macos_version: nil)
    if dependents
      formulae_have_untested_dependents?(
        formulae,
        platform:      platform,
        arch:          arch,
        macos_version: macos_version,
      )
    else
      return true if deleted_formulae.present?

      compatible_formulae = formulae.dup

      compatible_formulae.select! { |formula| formula.send(:"#{platform}_compatible?") }
      compatible_formulae.select! { |formula| formula.send(:"#{arch}_compatible?") }
      compatible_formulae.select! { |formula| formula.compatible_with?(macos_version) } if macos_version

      compatible_formulae.present?
    end
  end

  sig { void }
  def self.determine_test_runners
    args = determine_test_runners_args.parse

    eval_all = args.eval_all? || Homebrew::EnvConfig.eval_all?

    odie "`--dependents` requires `--eval-all` or `HOMEBREW_EVAL_ALL`!" if args.dependents? && !eval_all

    Formulary.enable_factory_cache!

    testing_formulae = args.named.first.split(",")
    testing_formulae.map! { |name| TestRunnerFormula.new(Formula[name], eval_all: eval_all) }
                    .freeze
    deleted_formulae = args.named.second&.split(",")

    runners = []

    linux_runner = ENV.fetch("HOMEBREW_LINUX_RUNNER") { raise "HOMEBREW_LINUX_RUNNER is not defined" }
    linux_cleanup = ENV.fetch("HOMEBREW_LINUX_CLEANUP") { raise "HOMEBREW_LINUX_CLEANUP is not defined" }

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

    if add_runner?(
      testing_formulae,
      platform:         :linux,
      arch:             :x86_64,
      deleted_formulae: deleted_formulae,
      dependents:       args.dependents?,
    )
      runners << linux_runner_spec
    end

    github_run_id = ENV.fetch("GITHUB_RUN_ID") { raise "GITHUB_RUN_ID is not defined" }
    github_run_attempt = ENV.fetch("GITHUB_RUN_ATTEMPT") { raise "GITHUB_RUN_ATTEMPT is not defined" }
    ephemeral_suffix = "-#{github_run_id}-#{github_run_attempt}"

    MacOSVersions::SYMBOLS.each_value do |version|
      macos_version = OS::Mac::Version.new(version)
      next if macos_version.outdated_release? || macos_version.prerelease?

      if add_runner?(
        testing_formulae,
        platform:         :macos,
        arch:             :x86_64,
        macos_version:    macos_version,
        deleted_formulae: deleted_formulae,
        dependents:       args.dependents?,
      )
        runners << { runner: "#{version}#{ephemeral_suffix}", cleanup: false }
      end

      next unless add_runner?(
        testing_formulae,
        platform:         :macos,
        arch:             :arm64,
        macos_version:    macos_version,
        deleted_formulae: deleted_formulae,
        dependents:       args.dependents?,
      )

      runner_name = "#{version}-arm64"
      # Use bare metal runner when testing dependents on Monterey.
      if macos_version >= :ventura || (macos_version >= :monterey && !args.dependents?)
        runners << { runner: "#{runner_name}#{ephemeral_suffix}", cleanup: false }
      elsif macos_version >= :big_sur
        runners << { runner: runner_name, cleanup: true }
      end
    end

    if !args.dependents? && runners.blank?
      # If there are no tests to run, add a runner that is meant to do nothing
      # to support making the `tests` job a required status check.
      runners << { runner: "ubuntu-latest", no_op: true }
    end

    github_output = ENV.fetch("GITHUB_OUTPUT") { raise "GITHUB_OUTPUT is not defined" }
    File.open(github_output, "a") do |f|
      f.puts("runners=#{runners.to_json}")
      f.puts("runners_present=#{runners.present?}")
    end
  end
end
