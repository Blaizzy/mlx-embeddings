# frozen_string_literal: true

require "cli/parser"
require "formula_installer"
require "install"
require "upgrade"
require "cask/cmd"
require "cask/utils"
require "cask/macos"

module Homebrew
  module_function

  def upgrade_args
    Homebrew::CLI::Parser.new do
      usage_banner <<~EOS
        `upgrade` [<options>] [<formula>|<cask>]

        Upgrade outdated casks and outdated, unpinned formulae using the same options they were originally
        installed with, plus any appended brew formula options. If <cask> or <formula> are specified,
        upgrade only the given <cask> or <formula> kegs (unless they are pinned; see `pin`, `unpin`).

        Unless `HOMEBREW_NO_INSTALL_CLEANUP` is set, `brew cleanup` will then be run for the
        upgraded formulae or, every 30 days, for all formulae.
      EOS
      switch "-d", "--debug",
             description: "If brewing fails, open an interactive debugging session with access to IRB "\
                          "or a shell inside the temporary build directory."
      switch "--formula",
             description: "Only upgrade outdated formulae."
      switch "--cask",
             description: "Only upgrade outdated casks."
      switch "-s", "--build-from-source",
             description: "Compile <formula> from source even if a bottle is available."
      switch "-i", "--interactive",
             description: "Download and patch <formula>, then open a shell. This allows the user to "\
                          "run `./configure --help` and otherwise determine how to turn the software "\
                          "package into a Homebrew package."
      switch "--force-bottle",
             description: "Install from a bottle if it exists for the current or newest version of "\
                          "macOS, even if it would not normally be used for installation."
      switch "--fetch-HEAD",
             description: "Fetch the upstream repository to detect if the HEAD installation of the "\
                          "formula is outdated. Otherwise, the repository's HEAD will only be checked for "\
                          "updates when a new stable or development version has been released."
      switch "--ignore-pinned",
             description: "Set a successful exit status even if pinned formulae are not upgraded."
      switch "--keep-tmp",
             description: "Retain the temporary files created during installation."
      switch "-f", "--force",
             description: "Install without checking for previously installed keg-only or "\
                          "non-migrated versions."
      switch "-v", "--verbose",
             description: "Print the verification and postinstall steps."
      switch "--display-times",
             env:         :display_install_times,
             description: "Print install times for each formula at the end of the run."
      switch "-n", "--dry-run",
             description: "Show what would be upgraded, but do not actually upgrade anything."
      switch "--greedy",
             description: "Upgrade casks with `auto_updates` or `version :latest`"
      conflicts "--build-from-source", "--force-bottle"
      conflicts "--formula", "--greedy"
      ["--formula", "-s", "--build-from-source", "-i", "--interactive",
       "--force-bottle", "--fetch-HEAD", "--ignore-pinned", "--keep-tmp",
       "--display-times"].each do |flag|
        conflicts "--cask", flag
      end
      formula_options
    end
  end

  def upgrade
    args = upgrade_args.parse

    formulae, casks = args.named.to_resolved_formulae_to_casks
    # If one or more formulae are specified, but no casks were
    # specified, we want to make note of that so we don't
    # try to upgrade all outdated casks.
    upgrade_formulae = formulae.present? && casks.blank?
    upgrade_casks = casks.present? && formulae.blank?

    upgrade_outdated_formulae(formulae, args: args) unless upgrade_casks
    upgrade_outdated_casks(casks, args: args) unless upgrade_formulae
  end

  def upgrade_outdated_formulae(formulae, args:)
    return if args.cask?

    FormulaInstaller.prevent_build_flags(args)

    Install.perform_preinstall_checks

    if formulae.blank?
      outdated = Formula.installed.select do |f|
        f.outdated?(fetch_head: args.fetch_HEAD?)
      end
    else
      outdated, not_outdated = formulae.partition do |f|
        f.outdated?(fetch_head: args.fetch_HEAD?)
      end

      not_outdated.each do |f|
        versions = f.installed_kegs.map(&:version)
        if versions.empty?
          ofail "#{f.full_specified_name} not installed"
        else
          version = versions.max
          opoo "#{f.full_specified_name} #{version} already installed"
        end
      end
    end

    return if outdated.blank?

    pinned = outdated.select(&:pinned?)
    outdated -= pinned
    formulae_to_install = outdated.map(&:latest_formula)

    if !pinned.empty? && !args.ignore_pinned?
      ofail "Not upgrading #{pinned.count} pinned #{"package".pluralize(pinned.count)}:"
      puts pinned.map { |f| "#{f.full_specified_name} #{f.pkg_version}" } * ", "
    end

    if formulae_to_install.empty?
      oh1 "No packages to upgrade"
    else
      verb = args.dry_run? ? "Would upgrade" : "Upgrading"
      oh1 "#{verb} #{formulae_to_install.count} outdated #{"package".pluralize(formulae_to_install.count)}:"
      formulae_upgrades = formulae_to_install.map do |f|
        if f.optlinked?
          "#{f.full_specified_name} #{Keg.new(f.opt_prefix).version} -> #{f.pkg_version}"
        else
          "#{f.full_specified_name} #{f.pkg_version}"
        end
      end
      puts formulae_upgrades.join("\n")
    end

    Upgrade.upgrade_formulae(formulae_to_install, args: args)

    Upgrade.check_installed_dependents(args: args)

    Homebrew.messages.display_messages(display_times: args.display_times?)
  end

  def upgrade_outdated_casks(casks, args:)
    return if args.formula?

    Cask::Cmd::Upgrade.upgrade_casks(
      *casks,
      force:          args.force?,
      greedy:         args.greedy?,
      dry_run:        args.dry_run?,
      binaries:       EnvConfig.cask_opts_binaries?,
      quarantine:     EnvConfig.cask_opts_quarantine?,
      require_sha:    EnvConfig.cask_opts_require_sha?,
      skip_cask_deps: args.skip_cask_deps?,
      verbose:        args.verbose?,
    )
  end
end
