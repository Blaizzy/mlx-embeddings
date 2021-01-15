# typed: true
# frozen_string_literal: true

require "formula"
require "cli/parser"
require "uninstall"

module Homebrew
  module_function

  def autoremove_args
    Homebrew::CLI::Parser.new do
      description <<~EOS
        Uninstall formulae that were only installed as a dependency of another formula and are now no longer needed.
      EOS
      switch "-n", "--dry-run",
             description: "List what would be uninstalled, but do not actually uninstall anything."

      named_args :none
    end
  end

  def get_removable_formulae(formulae)
    removable_formulae = Formula.installed_formulae_with_no_dependents(formulae).reject do |f|
      Tab.for_keg(f.any_installed_keg).installed_on_request
    end

    removable_formulae += get_removable_formulae(formulae - removable_formulae) if removable_formulae.present?

    removable_formulae
  end

  def autoremove
    args = autoremove_args.parse

    removable_formulae = get_removable_formulae(Formula.installed)
    return if removable_formulae.blank?

    formulae_names = removable_formulae.map(&:full_name).sort

    verb = args.dry_run? ? "Would uninstall" : "Uninstalling"
    oh1 "#{verb} #{formulae_names.count} unneeded #{"formula".pluralize(formulae_names.count)}:"
    puts formulae_names.join("\n")
    return if args.dry_run?

    kegs_by_rack = removable_formulae.map(&:any_installed_keg).group_by(&:rack)
    Uninstall.uninstall_kegs(kegs_by_rack)
  end
end
