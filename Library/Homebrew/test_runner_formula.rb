# typed: strict
# frozen_string_literal: true

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
