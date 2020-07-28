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
        `upgrade` [<options>] [<formula>]

        Upgrade outdated, unpinned formulae using the same options they were originally
        installed with, plus any appended brew formula options. If <formula> are specified,
        upgrade only the given <formula> kegs (unless they are pinned; see `pin`, `unpin`).

        Unless `HOMEBREW_NO_INSTALL_CLEANUP` is set, `brew cleanup` will then be run for the
        upgraded formulae or, every 30 days, for all formulae.
      EOS
      switch :debug,
             description: "If brewing fails, open an interactive debugging session with access to IRB "\
                          "or a shell inside the temporary build directory."
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
      switch :verbose,
             description: "Print the verification and postinstall steps."
      switch "--display-times",
             env:         :display_install_times,
             description: "Print install times for each formula at the end of the run."
      switch "-n", "--dry-run",
             description: "Show what would be upgraded, but do not actually upgrade anything."
      switch "--greedy",
             description: "Upgrade casks with `auto_updates` or `version :latest`"
      conflicts "--build-from-source", "--force-bottle"
      formula_options
    end
  end

  def upgrade
    args = upgrade_args.parse

    formulae, casks = args.resolved_formulae_casks
    # If one or more formulae are specified, but no casks were
    # specified, we want to make note of that so we don't
    # try to upgrade all outdated casks.
    named_formulae_specified = !formulae.empty? && casks.empty?
    named_casks_specified = !casks.empty? && formulae.empty?

    upgrade_outdated_formulae(formulae) unless named_casks_specified
    upgrade_outdated_casks(casks) unless named_formulae_specified
  end

  def upgrade_outdated_formulae(formulae)
    FormulaInstaller.prevent_build_flags unless DevelopmentTools.installed?

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

    upgrade_formulae(formulae_to_install, args: args)

    check_installed_dependents(args: args)

    Homebrew.messages.display_messages
  end

  def upgrade_outdated_casks(casks)
    cask_upgrade = Cask::Cmd::Upgrade.new(casks)
    cask_upgrade.force = args.force?
    cask_upgrade.dry_run = args.dry_run?
    cask_upgrade.greedy = args.greedy?
    cask_upgrade.run
  end
end
