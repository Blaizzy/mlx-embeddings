# typed: true
# frozen_string_literal: true

require "cask/cmd"
require "cask/cask_loader"
require "uninstall"

module Homebrew
  extend T::Sig

  module_function

  sig { returns(CLI::Parser) }
  def zap_args
    Homebrew::CLI::Parser.new do
      usage_banner <<~EOS
        `zap` [<options>] <formula>|<cask>

        Remove all files associated with the given <formula> or <cask>.
        Implicitly also performs all actions associated with `uninstall`.

        *May remove files which are shared between applications.*
      EOS
      switch "-f", "--force",
             description: "Delete all installed versions of <formula>. Uninstall even if <cask> is not " \
                          "installed, overwrite existing files and ignore errors when removing files."
      switch "--ignore-dependencies",
             description: "Don't fail uninstall, even if <formula> is a dependency of any installed "\
                          "formulae."

      switch "--formula", "--formulae",
             description: "Treat all named arguments as formulae."
      switch "--cask", "--casks",
             description: "Treat all named arguments as casks."
      conflicts "--formula", "--cask"

      min_named :formula_or_cask
    end
  end

  def zap
    args = zap_args.parse

    only = :formula if args.formula? && !args.cask?
    only = :cask if args.cask? && !args.formula?

    all_kegs, casks = args.named.to_kegs_to_casks(only: only, ignore_unavailable: args.force?, all_kegs: args.force?)
    kegs_by_rack = all_kegs.group_by(&:rack)

    Uninstall.uninstall_kegs(
      kegs_by_rack,
      force:               args.force?,
      ignore_dependencies: args.ignore_dependencies?,
      named_args:          args.named,
    )

    Cask::Cmd::Zap.zap_casks(
      *casks,
      binaries: EnvConfig.cask_opts_binaries?,
      verbose:  args.verbose?,
      force:    args.force?,
    )
  end
end
