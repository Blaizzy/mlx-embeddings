# typed: false
# frozen_string_literal: true

require "formula"
require "cli/parser"
require "uninstall"

module Homebrew
  extend Uninstall

  module_function

  def autoremove_args
    Homebrew::CLI::Parser.new do
      usage_banner <<~EOS
        `autoremove` [<options>]

        Remove packages that weren't installed on request and are no longer needed.
      EOS
      switch "-n", "--dry-run",
             description: "Just print what would be removed."
      named 0
    end
  end

  def get_removable_formulae(formulae)
    removable_formulae = Formula.installed_non_deps(formulae).reject do |f|
      Tab.for_keg(f.any_installed_keg).installed_on_request
    end

    removable_formulae += get_removable_formulae(formulae - removable_formulae) if removable_formulae.any?

    removable_formulae
  end

  def autoremove
    args = autoremove_args.parse

    removable_formulae = get_removable_formulae(Formula.installed)

    return if removable_formulae.blank?

    formulae_names = removable_formulae.map(&:full_name).sort

    oh1 "Formulae that could be removed"
    puts formulae_names

    return if args.dry_run?

    kegs_by_rack = removable_formulae.map(&:any_installed_keg).group_by(&:rack)
    uninstall_kegs(kegs_by_rack)
  end
end
