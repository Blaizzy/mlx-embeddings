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

  sig { params(formula: Formula).void }
  def initialize(formula)
    @formula = T.let(formula, Formula)
    @name = T.let(formula.name, String)
    @dependent_hash = T.let({}, T::Hash[Symbol, T::Array[TestRunnerFormula]])
    freeze
  end

  sig { returns(T::Boolean) }
  def macos_only?
    formula.requirements.any? { |r| r.is_a?(MacOSRequirement) && !r.version_specified? }
  end

  sig { returns(T::Boolean) }
  def linux_only?
    formula.requirements.any?(LinuxRequirement)
  end

  sig { returns(T::Boolean) }
  def x86_64_only?
    formula.requirements.any? { |r| r.is_a?(ArchRequirement) && (r.arch == :x86_64) }
  end

  sig { returns(T::Boolean) }
  def arm64_only?
    formula.requirements.any? { |r| r.is_a?(ArchRequirement) && (r.arch == :arm64) }
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

  sig { params(cache_key: Symbol).returns(T::Array[TestRunnerFormula]) }
  def dependents(cache_key)
    # TODO: Check that `--eval-all` or `HOMEBREW_EVAL_ALL` is set.
    @dependent_hash[cache_key] ||= with_env(HOMEBREW_EVAL_ALL: "1") do
      Formula.all
             .select { |candidate_f| candidate_f.deps.map(&:name).include?(name) }
             .map { |f| TestRunnerFormula.new(f) }
             .freeze
    end

    @dependent_hash.fetch(cache_key)
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
      switch "--dependents",
             description: "Determine runners for testing dependents."

      named_args min: 1, max: 2

      hide_from_man_page!
    end
  end
  sig {
    params(
      testing_formulae:     T::Array[TestRunnerFormula],
      reject_platform:      T.nilable(Symbol),
      reject_arch:          T.nilable(Symbol),
      select_macos_version: T.nilable(OS::Mac::Version),
    ).returns(T::Boolean)
  }
  def self.formulae_have_untested_dependents?(testing_formulae, reject_platform:, reject_arch:, select_macos_version:)
    testing_formulae.any? do |formula|
      # If the formula has a platform/arch/macOS version requirement, then its
      # dependents don't need to be tested if these requirements are not satisfied.
      next false if reject_platform && formula.send(:"#{reject_platform}_only?")
      next false if reject_arch && formula.send(:"#{reject_arch}_only?")
      next false if select_macos_version && !formula.compatible_with?(select_macos_version)

      compatible_dependents = formula.dependents(:"#{reject_platform}_#{reject_arch}_#{select_macos_version}").dup

      compatible_dependents.reject! { |dependent_f| dependent_f.send(:"#{reject_arch}_only?") } if reject_arch
      compatible_dependents.reject! { |dependent_f| dependent_f.send(:"#{reject_platform}_only?") } if reject_platform
      if select_macos_version
        compatible_dependents.select! { |dependent_f| dependent_f.compatible_with?(select_macos_version) }
      end

      (compatible_dependents - testing_formulae).present?
    end
  end

  sig {
    params(
      formulae:             T::Array[TestRunnerFormula],
      dependents:           T::Boolean,
      deleted_formulae:     T.nilable(T::Array[String]),
      reject_platform:      T.nilable(Symbol),
      reject_arch:          T.nilable(Symbol),
      select_macos_version: T.nilable(OS::Mac::Version),
    ).returns(T::Boolean)
  }
  def self.add_runner?(formulae,
                       dependents:,
                       deleted_formulae:,
                       reject_platform: nil,
                       reject_arch: nil,
                       select_macos_version: nil)
    if dependents
      formulae_have_untested_dependents?(
        formulae,
        reject_platform:      reject_platform,
        reject_arch:          reject_arch,
        select_macos_version: select_macos_version,
      )
    else
      return true if deleted_formulae.present?

      compatible_formulae = formulae.dup

      compatible_formulae.reject! { |formula| formula.send(:"#{reject_platform}_only?") } if reject_platform
      compatible_formulae.reject! { |formula| formula.send(:"#{reject_arch}_only?") } if reject_arch
      compatible_formulae.select! { |formula| formula.compatible_with?(select_macos_version) } if select_macos_version

      compatible_formulae.present?
    end
  end

  sig { void }
  def self.determine_test_runners
    args = determine_test_runners_args.parse

    Formulary.enable_factory_cache!

    testing_formulae = args.named.first.split(",")
    testing_formulae.map! { |name| TestRunnerFormula.new(Formula[name]) }
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

    if args.dependents?
      Homebrew::SimulateSystem.os = :linux
      Homebrew::SimulateSystem.arch = :intel
      Formulary.clear_cache
    end

    if add_runner?(
      testing_formulae,
      reject_platform:  :macos,
      reject_arch:      :arm64,
      deleted_formulae: deleted_formulae,
      dependents:       args.dependents?,
    )
      runners << linux_runner_spec
    end

    github_run_id = ENV.fetch("GITHUB_RUN_ID") { raise "GITHUB_RUN_ID is not defined" }
    github_run_attempt = ENV.fetch("GITHUB_RUN_ATTEMPT") { raise "GITHUB_RUN_ATTEMPT is not defined" }
    ephemeral_suffix = "-#{github_run_id}-#{github_run_attempt}"

    MacOSVersions::SYMBOLS.each do |symbol, version|
      macos_version = OS::Mac::Version.new(version)
      next if macos_version.outdated_release? || macos_version.prerelease?

      if args.dependents?
        Formulary.clear_cache
        Homebrew::SimulateSystem.os = symbol
        Homebrew::SimulateSystem.arch = :intel
      end

      if add_runner?(
        testing_formulae,
        reject_platform:      :linux,
        reject_arch:          :arm64,
        select_macos_version: macos_version,
        deleted_formulae:     deleted_formulae,
        dependents:           args.dependents?,
      )
        runners << { runner: "#{version}#{ephemeral_suffix}", cleanup: false }
      end

      if args.dependents?
        Formulary.clear_cache
        Homebrew::SimulateSystem.os = symbol
        Homebrew::SimulateSystem.arch = :arm
      end

      next unless add_runner?(
        testing_formulae,
        reject_platform:      :linux,
        reject_arch:          :x86_64,
        select_macos_version: macos_version,
        deleted_formulae:     deleted_formulae,
        dependents:           args.dependents?,
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
