# frozen_string_literal: true

require "formula"
require "cli/parser"

module Homebrew
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

  def get_removable_formulae(installed_formulae)
    removable_formulae = []

    installed_formulae.each do |formula|
      # Reject formulae installed on request.
      next if Tab.for_keg(formula.any_installed_keg).installed_on_request
      # Reject formulae which are needed at runtime by other formulae.
      next if installed_formulae.flat_map(&:runtime_formula_dependencies).include?(formula)

      removable_formulae << installed_formulae.delete(formula)
      removable_formulae += get_removable_formulae(installed_formulae)
    end

    removable_formulae
  end

  def autoremove
    args = autoremove_args.parse

    removable_formulae = get_removable_formulae(Formula.installed.sort)

    return if removable_formulae.blank?

    formulae_names = removable_formulae.map(&:full_name)

    oh1 "Formulae that could be removed"
    puts formulae_names

    return if args.dry_run?

    system HOMEBREW_BREW_FILE, "rm", *formulae_names
  end
end
