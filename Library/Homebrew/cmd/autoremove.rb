# typed: true
# frozen_string_literal: true

require "autoremove"
require "cli/parser"

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

    Autoremove.remove_unused_formulae(dry_run: args.dry_run?)
  end
end
