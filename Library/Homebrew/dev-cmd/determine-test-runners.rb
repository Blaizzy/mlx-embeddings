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

  sig { params(name: String).void }
  def initialize(name)
    @name = T.let(name, String)
    @formula = T.let(Formula[name], Formula)
    @dependent_hash = T.let({}, T::Hash[T::Boolean, T::Array[TestRunnerFormula]])
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

  sig { params(macos_version: MacOS::Version).returns(T::Boolean) }
  def compatible_with?(macos_version)
    # Assign to a variable to assist type-checking.
    requirement = versioned_macos_requirement
    return true if requirement.blank?

    macos_version.public_send(requirement.comparator, requirement.version)
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
             description: "Determine runners for testing dependents. (requires Linux)"

      named_args min: 1, max: 2

      hide_from_man_page!
    end
  end

  sig {
    params(
      _testing_formulae:    T::Array[TestRunnerFormula],
      reject_platform:      T.nilable(Symbol),
      reject_arch:          T.nilable(Symbol),
      select_macos_version: T.nilable(MacOS::Version),
    ).void
  }
  def self.formulae_have_untested_dependents?(_testing_formulae, reject_platform:,
                                              reject_arch:, select_macos_version:)
    odie "`--dependents` is supported only on Linux!"
  end

  sig {
    params(
      formulae:             T::Array[TestRunnerFormula],
      dependents:           T::Boolean,
      deleted_formulae:     T.nilable(T::Array[String]),
      reject_platform:      T.nilable(Symbol),
      reject_arch:          T.nilable(Symbol),
      select_macos_version: T.nilable(MacOS::Version),
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
    testing_formulae = args.named.first.split(",")
    testing_formulae.map! { |name| TestRunnerFormula.new(name) }
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

    with_env(HOMEBREW_SIMULATE_MACOS_ON_LINUX: nil) do
      if add_runner?(
        testing_formulae,
        reject_platform:  :macos,
        reject_arch:      :arm64,
        deleted_formulae: deleted_formulae,
        dependents:       args.dependents?,
      )
        runners << linux_runner_spec
      end
    end

    # TODO: `HOMEBREW_SIMULATE_MACOS_ON_LINUX` simulates the oldest version of macOS.
    #       Handle formulae that are dependents only on new versions of macOS.
    with_env(HOMEBREW_SIMULATE_MACOS_ON_LINUX: "1") do
      if add_runner?(
        testing_formulae,
        reject_platform:  :linux,
        deleted_formulae: deleted_formulae,
        dependents:       args.dependents?,
      )
        add_intel_runners = add_runner?(
          testing_formulae,
          reject_platform:  :linux,
          reject_arch:      :arm64,
          deleted_formulae: deleted_formulae,
          dependents:       args.dependents?,
        )
        add_m1_runners = add_runner?(
          testing_formulae,
          reject_platform:  :linux,
          reject_arch:      :x86_64,
          deleted_formulae: deleted_formulae,
          dependents:       args.dependents?,
        )

        github_run_id = ENV.fetch("GITHUB_RUN_ID") { raise "GITHUB_RUN_ID is not defined" }
        github_run_attempt = ENV.fetch("GITHUB_RUN_ATTEMPT") { raise "GITHUB_RUN_ATTEMPT is not defined" }

        MacOSVersions::SYMBOLS.each_value do |version|
          macos_version = MacOS::Version.new(version)
          next if macos_version.outdated_release? || macos_version.prerelease?

          unless add_runner?(
            testing_formulae,
            reject_platform:      :linux,
            select_macos_version: macos_version,
            deleted_formulae:     deleted_formulae,
            dependents:           args.dependents?,
          )
            next # No formulae to test on this macOS version.
          end

          ephemeral_suffix = "-#{github_run_id}-#{github_run_attempt}"
          runners << { runner: "#{macos_version}#{ephemeral_suffix}", cleanup: false } if add_intel_runners

          next unless add_m1_runners

          # Use bare metal runner when testing dependents on Monterey.
          if macos_version >= :ventura || (macos_version >= :monterey && !args.dependents?)
            runners << { runner: "#{macos_version}-arm64#{ephemeral_suffix}", cleanup: false }
          elsif macos_version >= :big_sur
            runners << { runner: "#{macos_version}-arm64", cleanup: true }
          end
        end
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

require "extend/os/dev-cmd/determine-test-runners"
