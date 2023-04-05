# typed: strict
# frozen_string_literal: true

require "test_runner_formula"

class LinuxRunnerSpec < T::Struct
  extend T::Sig

  const :name, String
  const :runner, String
  const :container, T::Hash[Symbol, String]
  const :workdir, String
  const :timeout, Integer
  const :cleanup, T::Boolean

  sig {
    returns({
      name:      String,
      runner:    String,
      container: T::Hash[Symbol, String],
      workdir:   String,
      timeout:   Integer,
      cleanup:   T::Boolean,
    })
  }
  def to_h
    {
      name:      name,
      runner:    runner,
      container: container,
      workdir:   workdir,
      timeout:   timeout,
      cleanup:   cleanup,
    }
  end
end

class MacOSRunnerSpec < T::Struct
  extend T::Sig

  const :name, String
  const :runner, String
  const :cleanup, T::Boolean

  sig { returns({ name: String, runner: String, cleanup: T::Boolean }) }
  def to_h
    {
      name:    name,
      runner:  runner,
      cleanup: cleanup,
    }
  end
end

class GitHubRunner < T::Struct
  const :platform, Symbol
  const :arch, Symbol
  const :spec, T.any(LinuxRunnerSpec, MacOSRunnerSpec)
  const :macos_version, T.nilable(OS::Mac::Version)
end

class GitHubRunnerMatrix
  extend T::Sig

  # FIXME: Enable cop again when https://github.com/sorbet/sorbet/issues/3532 is fixed.
  # rubocop:disable Style/MutableConstant
  MaybeStringArray = T.type_alias { T.nilable(T::Array[String]) }
  private_constant :MaybeStringArray

  RunnerSpec = T.type_alias { T.any(LinuxRunnerSpec, MacOSRunnerSpec) }
  private_constant :RunnerSpec
  # rubocop:enable Style/MutableConstant

  sig { returns(T::Array[GitHubRunner]) }
  attr_reader :available_runners

  sig {
    params(
      testing_formulae: T::Array[TestRunnerFormula],
      deleted_formulae: MaybeStringArray,
      dependent_matrix: T::Boolean,
    ).void
  }
  def initialize(testing_formulae, deleted_formulae, dependent_matrix:)
    @testing_formulae = T.let(testing_formulae, T::Array[TestRunnerFormula])
    @deleted_formulae = T.let(deleted_formulae, MaybeStringArray)
    @dependent_matrix = T.let(dependent_matrix, T::Boolean)

    @available_runners = T.let([], T::Array[GitHubRunner])
    generate_available_runners!

    @active_runners = T.let(
      @available_runners.select { |runner| active_runner?(runner) },
      T::Array[GitHubRunner],
    )

    freeze
  end

  sig { returns(T::Array[T::Hash[Symbol, T.untyped]]) }
  def active_runner_specs_hash
    @active_runners.map(&:spec)
                   .map(&:to_h)
  end

  sig { returns(LinuxRunnerSpec) }
  def linux_runner_spec
    linux_runner  = ENV.fetch("HOMEBREW_LINUX_RUNNER")
    linux_cleanup = ENV.fetch("HOMEBREW_LINUX_CLEANUP")

    LinuxRunnerSpec.new(
      name:      "Linux",
      runner:    linux_runner,
      container: {
        image:   "ghcr.io/homebrew/ubuntu22.04:master",
        options: "--user=linuxbrew -e GITHUB_ACTIONS_HOMEBREW_SELF_HOSTED",
      },
      workdir:   "/github/home",
      timeout:   4320,
      cleanup:   linux_cleanup == "true",
    )
  end

  sig { void }
  def generate_available_runners!
    @available_runners << GitHubRunner.new(platform: :linux, arch: :x86_64, spec: linux_runner_spec)

    github_run_id      = ENV.fetch("GITHUB_RUN_ID")
    github_run_attempt = ENV.fetch("GITHUB_RUN_ATTEMPT")
    ephemeral_suffix = "-#{github_run_id}-#{github_run_attempt}"

    MacOSVersions::SYMBOLS.each_value do |version|
      macos_version = OS::Mac::Version.new(version)
      next if macos_version.outdated_release? || macos_version.prerelease?

      spec = MacOSRunnerSpec.new(
        name:    "macOS #{version}-x86_64",
        runner:  "#{version}#{ephemeral_suffix}",
        cleanup: false,
      )
      @available_runners << GitHubRunner.new(
        platform:      :macos,
        arch:          :x86_64,
        spec:          spec,
        macos_version: macos_version,
      )

      next unless macos_version >= :big_sur

      # Use bare metal runner when testing dependents on ARM64 Monterey.
      runner, cleanup = if (macos_version >= :ventura && @dependent_matrix) || macos_version >= :monterey
        ["#{version}-arm64#{ephemeral_suffix}", false]
      else
        ["#{version}-arm64", true]
      end

      spec = MacOSRunnerSpec.new(name: "macOS #{version}-arm64", runner: runner, cleanup: cleanup)
      @available_runners << GitHubRunner.new(
        platform:      :macos,
        arch:          :arm64,
        spec:          spec,
        macos_version: macos_version,
      )
    end
  end

  sig { params(runner: GitHubRunner).returns(T::Boolean) }
  def active_runner?(runner)
    if @dependent_matrix
      formulae_have_untested_dependents?(runner)
    else
      return true if @deleted_formulae.present?

      compatible_formulae = @testing_formulae.dup

      platform = runner.platform
      arch = runner.arch
      macos_version = runner.macos_version

      compatible_formulae.select! { |formula| formula.send(:"#{platform}_compatible?") }
      compatible_formulae.select! { |formula| formula.send(:"#{arch}_compatible?") }
      compatible_formulae.select! { |formula| formula.compatible_with?(macos_version) } if macos_version

      compatible_formulae.present?
    end
  end

  sig { params(runner: GitHubRunner).returns(T::Boolean) }
  def formulae_have_untested_dependents?(runner)
    platform = runner.platform
    arch = runner.arch
    macos_version = runner.macos_version

    @testing_formulae.any? do |formula|
      # If the formula has a platform/arch/macOS version requirement, then its
      # dependents don't need to be tested if these requirements are not satisfied.
      next false unless formula.send(:"#{platform}_compatible?")
      next false unless formula.send(:"#{arch}_compatible?")
      next false if macos_version.present? && !formula.compatible_with?(macos_version)

      compatible_dependents = formula.dependents(platform: platform, arch: arch, macos_version: macos_version&.to_sym)
                                     .dup

      compatible_dependents.select! { |dependent_f| dependent_f.send(:"#{platform}_compatible?") }
      compatible_dependents.select! { |dependent_f| dependent_f.send(:"#{arch}_compatible?") }
      compatible_dependents.select! { |dependent_f| dependent_f.compatible_with?(macos_version) } if macos_version

      # These arrays will generally have been generated by different Formulary caches,
      # so we can only compare them by name and not directly.
      (compatible_dependents.map(&:name) - @testing_formulae.map(&:name)).present?
    end
  end
end
