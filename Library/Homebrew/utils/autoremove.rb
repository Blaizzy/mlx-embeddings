# typed: true
# frozen_string_literal: true

module Utils
  # Helper function for finding autoremovable formulae.
  #
  # @private
  module Autoremove
    module_function

    # An array of all installed {Formula} with {Cask} dependents.
    # @private
    def formulae_with_cask_dependents(casks)
      casks.flat_map { |cask| cask.depends_on[:formula] }
           .compact
           .map { |f| Formula[f] }
           .flat_map { |f| [f, *f.runtime_formula_dependencies].compact }
    end
    private_class_method :formulae_with_cask_dependents

    # An array of all installed {Formula} without runtime {Formula}
    # dependents for bottles and without build {Formula} dependents
    # for those built from source.
    # @private
    def formulae_with_no_formula_dependents(formulae)
      return [] if formulae.blank?

      dependents = T.let([], T::Array[Formula])
      formulae.each do |formula|
        dependents += formula.runtime_formula_dependencies

        # Ignore build dependencies when the formula is a bottle
        next if Tab.for_keg(formula.any_installed_keg).poured_from_bottle

        formula.deps.select(&:build?).each do |dep|
          dependents << dep.to_formula
        rescue FormulaUnavailableError
          # do nothing
        end
      end
      formulae - dependents
    end
    private_class_method :formulae_with_no_formula_dependents

    # Recursive function that returns an array of {Formula} without
    # {Formula} dependents that weren't installed on request.
    # @private
    def unused_formulae_with_no_formula_dependents(formulae)
      unused_formulae = formulae_with_no_formula_dependents(formulae).reject do |f|
        Tab.for_keg(f.any_installed_keg).installed_on_request
      end

      if unused_formulae.present?
        unused_formulae += unused_formulae_with_no_formula_dependents(formulae - unused_formulae)
      end

      unused_formulae
    end
    private_class_method :unused_formulae_with_no_formula_dependents

    # An array of {Formula} without {Formula} or {Cask}
    # dependents that weren't installed on request and without
    # build dependencies for {Formula} installed from source.
    # @private
    def removable_formulae(formulae, casks)
      unused_formulae = unused_formulae_with_no_formula_dependents(formulae)
      unused_formulae - formulae_with_cask_dependents(casks)
    end
  end
end
