# typed: strict
# frozen_string_literal: true

class TestRunnerFormula
  extend T::Sig

  sig { returns(T::Array[TestRunnerFormula]) }
  def dependents
    @dependent_hash[ENV["HOMEBREW_SIMULATE_MACOS_ON_LINUX"].present?] ||= with_env(HOMEBREW_STDERR: "1") do
      Utils.safe_popen_read(
        HOMEBREW_BREW_FILE, "uses", "--formulae", "--eval-all", "--include-build", "--include-test", name
      ).split("\n").map { |dependent| TestRunnerFormula.new(dependent) }.freeze
    end

    @dependent_hash.fetch(ENV["HOMEBREW_SIMULATE_MACOS_ON_LINUX"].present?)
  end
end

module Homebrew
  extend T::Sig

  sig {
    params(
      testing_formulae:     T::Array[TestRunnerFormula],
      reject_platform:      T.nilable(Symbol),
      reject_arch:          T.nilable(Symbol),
      select_macos_version: T.nilable(MacOS::Version),
    ).returns(T::Boolean)
  }
  def self.formulae_have_untested_dependents?(testing_formulae, reject_platform:, reject_arch:, select_macos_version:)
    testing_formulae.any? do |formula|
      # If the formula has a platform/arch/macOS version requirement, then its
      # dependents don't need to be tested if these requirements are not satisfied.
      next false if reject_platform && formula.send(:"#{reject_platform}_only?")
      next false if reject_arch && formula.send(:"#{reject_arch}_only?")
      next false if select_macos_version && !formula.compatible_with?(select_macos_version)

      compatible_dependents = formula.dependents.dup

      compatible_dependents.reject! { |dependent_f| dependent_f.send(:"#{reject_arch}_only?") } if reject_arch
      compatible_dependents.reject! { |dependent_f| dependent_f.send(:"#{reject_platform}_only?") } if reject_platform
      if select_macos_version
        compatible_dependents.select! { |dependent_f| dependent_f.compatible_with?(select_macos_version) }
      end

      (compatible_dependents - testing_formulae).present?
    end
  end
end
