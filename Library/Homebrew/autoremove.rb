# typed: false
# frozen_string_literal: true

require "cask/caskroom"
require "formula"
require "uninstall"

module Homebrew
  # Helpers for removing unused formulae.
  #
  # @api private
  module Autoremove
    module_function

    def remove_unused_formulae(dry_run: false)
      removable_formulae = unused_formulae_with_no_dependents

      return if removable_formulae.blank?

      formulae_names = removable_formulae.map(&:full_name).sort

      verb = dry_run ? "Would autoremove" : "Autoremoving"
      oh1 "#{verb} #{formulae_names.count} unneeded #{"formula".pluralize(formulae_names.count)}:"
      puts formulae_names.join("\n")
      return if dry_run

      kegs_by_rack = removable_formulae.map(&:any_installed_keg).group_by(&:rack)
      Uninstall.uninstall_kegs(kegs_by_rack)
    end

    # An array of installed {Formula} without {Formula} or {Cask}
    # dependents that weren't installed on request.
    # @private
    def unused_formulae_with_no_dependents
      unused_formulae = unused_formulae_with_no_formula_dependents(Formula.installed)
      unused_formulae - installed_formulae_with_cask_dependents
    end

    # Recursive function that returns an array of installed {Formula} without
    # {Formula} dependents that weren't installed on request.
    # @private
    def unused_formulae_with_no_formula_dependents(formulae)
      unused_formulae = Formula.installed_formulae_with_no_dependents(formulae).reject do |f|
        Tab.for_keg(f.any_installed_keg).installed_on_request
      end

      if unused_formulae.present?
        unused_formulae += unused_formulae_with_no_formula_dependents(formulae - unused_formulae)
      end

      unused_formulae
    end

    # An array of all installed {Formula} with {Cask} dependents.
    # @private
    def installed_formulae_with_cask_dependents
      Cask::Caskroom.casks
                    .flat_map { |cask| cask.depends_on[:formula] }
                    .compact
                    .map { |f| Formula[f] }
                    .flat_map { |f| [f, *f.runtime_formula_dependencies].compact }
    end
  end
end
