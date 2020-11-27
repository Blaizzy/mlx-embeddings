# typed: true
# frozen_string_literal: true

require "keg"
require "formula"
require "diagnostic"
require "migrator"
require "cli/parser"
require "cask/cmd"
require "cask/cask_loader"
require "uninstall"

module Homebrew
  extend T::Sig

  module_function

  sig { returns(CLI::Parser) }
  def uninstall_args
    Homebrew::CLI::Parser.new do
      description <<~EOS
        Uninstall a <formula> or <cask>.
      EOS
      switch "-f", "--force",
             description: "Delete all installed versions of <formula>. Uninstall even if <cask> is not " \
                          "installed, overwrite existing files and ignore errors when removing files."
      switch "--zap",
             description: "Remove all files associated with a <cask>. " \
                          "*May remove files which are shared between applications.*"
      switch "--ignore-dependencies",
             description: "Don't fail uninstall, even if <formula> is a dependency of any installed "\
                          "formulae."
      switch "--formula", "--formulae",
             description: "Treat all named arguments as formulae."
      switch "--cask", "--casks",
             description: "Treat all named arguments as casks."

      conflicts "--formula", "--cask"
      conflicts "--formula", "--zap"

      named_args [:installed_formula, :installed_cask], min: 1
    end
  end

  def uninstall
    args = uninstall_args.parse

    all_kegs, casks = args.named.to_kegs_to_casks(
      ignore_unavailable: args.force?,
      all_kegs:           args.force?,
    )

    kegs_by_rack = all_kegs.group_by(&:rack)

    Uninstall.uninstall_kegs(
      kegs_by_rack,
      force:               args.force?,
      ignore_dependencies: args.ignore_dependencies?,
      named_args:          args.named,
    )

    if args.zap?
      T.unsafe(Cask::Cmd::Zap).zap_casks(
        *casks,
        verbose: args.verbose?,
        force:   args.force?,
      )
    else
      T.unsafe(Cask::Cmd::Uninstall).uninstall_casks(
        *casks,
        binaries: EnvConfig.cask_opts_binaries?,
        verbose:  args.verbose?,
        force:    args.force?,
      )
    end
  end
end
