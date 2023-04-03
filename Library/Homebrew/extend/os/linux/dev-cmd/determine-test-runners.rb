# typed: false
# frozen_string_literal: true

require "formula"

class TestRunnerFormula
  attr_reader :name, :formula

  def initialize(name)
    @name = name
    @formula = Formula[name]
    freeze
  end

  def macos_only?
    formula.requirements.any? { |r| r.is_a?(MacOSRequirement) && !r.version_specified? }
  end

  def linux_only?
    formula.requirements.any?(LinuxRequirement)
  end

  def x86_64_only?
    formula.requirements.any? { |r| r.is_a?(ArchRequirement) && (r.arch == :x86_64) }
  end

  def arm64_only?
    formula.requirements.any? { |r| r.is_a?(ArchRequirement) && (r.arch == :arm64) }
  end

  def versioned_macos_requirement
    formula.requirements.find { |r| r.is_a?(MacOSRequirement) && r.version_specified? }
  end

  def compatible_with?(macos_version)
    return true if versioned_macos_requirement.blank?

    macos_version.public_send(versioned_macos_requirement.comparator, versioned_macos_requirement.version)
  end

  def dependents
    @dependent_hash ||= {}
    @dependent_hash[ENV["HOMEBREW_SIMULATE_MACOS_ON_LINUX"].present?] ||= with_env(HOMEBREW_STDERR: "1") do
      Utils.safe_popen_read(
        HOMEBREW_BREW_FILE, "uses", "--formulae", "--eval-all", "--include-build", "--include-test", name
      ).split("\n").map { |dependent| TestRunnerFormula.new(dependent) }.freeze
    end

    @dependent_hash[ENV["HOMEBREW_SIMULATE_MACOS_ON_LINUX"].present?]
  end
end

module Homebrew
  module_function

  def formulae_have_untested_dependents?(testing_formulae, reject_platform:, reject_arch:, select_macos_version:)
    testing_formulae.any? do |formula|
      # If the formula has a platform/arch/macOS version requirement, then its
      # dependents don't need to be tested if these requirements are not satisfied.
      next false if reject_platform && formula.method("#{reject_platform}_only?").call
      next false if reject_arch && formula.method("#{reject_arch}_only?").call
      next false if select_macos_version && !formula.compatible_with?(select_macos_version)

      compatible_dependents = formula.dependents.dup

      compatible_dependents.reject! { |dependent_f| dependent_f.method("#{reject_arch}_only?").call } if reject_arch

      if reject_platform
        compatible_dependents.reject! { |dependent_f| dependent_f.method("#{reject_platform}_only?").call }
      end

      if select_macos_version
        compatible_dependents.select! { |dependent_f| dependent_f.compatible_with?(select_macos_version) }
      end

      (compatible_dependents - testing_formulae).present?
    end
  end

  def add_runner?(formulae,
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
      compatible_formulae.reject! { |formula| formula.method("#{reject_platform}_only?").call } if reject_platform
      compatible_formulae.reject! { |formula| formula.method("#{reject_arch}_only?").call } if reject_arch
      compatible_formulae.select! { |formula| formula.compatible_with?(select_macos_version) } if select_macos_version

      compatible_formulae.present?
    end
  end

  def determine_test_runners
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

          ephemeral_suffix = "-#{ENV.fetch("GITHUB_RUN_ID")}-#{ENV.fetch("GITHUB_RUN_ATTEMPT")}"
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
