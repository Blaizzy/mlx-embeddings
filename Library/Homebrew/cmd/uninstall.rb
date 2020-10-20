# typed: false
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
      usage_banner <<~EOS
        `uninstall`, `rm`, `remove` [<options>] <formula>

        Uninstall <formula>.
      EOS
      switch "-f", "--force",
             description: "Delete all installed versions of <formula>."
      switch "--ignore-dependencies",
             description: "Don't fail uninstall, even if <formula> is a dependency of any installed "\
                          "formulae."

      min_named :formula
    end
  end

  def uninstall
    args = uninstall_args.parse

    if args.force?
      casks = []
      kegs_by_rack = {}

      args.named.each do |name|
        rack = Formulary.to_rack(name)

        if rack.directory?
          kegs_by_rack[rack] = rack.subdirs.map { |d| Keg.new(d) }
        else
          begin
            casks << Cask::CaskLoader.load(name)
          rescue Cask::CaskUnavailableError
            # Since the uninstall was forced, ignore any unavailable casks
          end
        end
      end
    else
      all_kegs, casks = args.named.to_kegs_to_casks
      kegs_by_rack = all_kegs.group_by(&:rack)
    end

    Uninstall.uninstall_kegs(kegs_by_rack,
                             force:               args.force?,
                             ignore_dependencies: args.ignore_dependencies?,
                             named_args:          args.named)

    return if casks.blank?

    Cask::Cmd::Uninstall.uninstall_casks(
      *casks,
      binaries: EnvConfig.cask_opts_binaries?,
      verbose:  args.verbose?,
      force:    args.force?,
    )
  rescue MultipleVersionsInstalledError => e
    ofail e
  ensure
    # If we delete Cellar/newname, then Cellar/oldname symlink
    # can become broken and we have to remove it.
    if HOMEBREW_CELLAR.directory?
      HOMEBREW_CELLAR.children.each do |rack|
        rack.unlink if rack.symlink? && !rack.resolved_path_exists?
      end
    end
  end
end
