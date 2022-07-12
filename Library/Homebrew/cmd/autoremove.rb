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

  def autoremove
    args = autoremove_args.parse

    Uninstall.autoremove_kegs(Formula.removable_formulae, dry_run: args.dry_run?)
  end
end
