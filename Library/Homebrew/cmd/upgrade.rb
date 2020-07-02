# frozen_string_literal: true

require "cli/parser"
require "formula_installer"
require "install"
require "upgrade"

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
      switch :force,
             description: "Install without checking for previously installed keg-only or "\
                          "non-migrated versions."
      switch :verbose,
             description: "Print the verification and postinstall steps."
      switch "--display-times",
             env:         :display_install_times,
             description: "Print install times for each formula at the end of the run."
      switch "-n", "--dry-run",
             description: "Show what would be upgraded, but do not actually upgrade anything."
      conflicts "--build-from-source", "--force-bottle"
      formula_options
    end
  end

  def upgrade
    upgrade_args.parse

    FormulaInstaller.prevent_build_flags unless DevelopmentTools.installed?

    Install.perform_preinstall_checks

    if args.no_named?
      outdated = Formula.installed.select do |f|
        f.outdated?(fetch_head: args.fetch_HEAD?)
      end

      exit 0 if outdated.empty?
    else
      outdated = args.resolved_formulae.select do |f|
        f.outdated?(fetch_head: args.fetch_HEAD?)
      end

      (args.resolved_formulae - outdated).each do |f|
        versions = f.installed_kegs.map(&:version)
        if versions.empty?
          ofail "#{f.full_specified_name} not installed"
        else
          version = versions.max
          opoo "#{f.full_specified_name} #{version} already installed"
        end
      end
      return if outdated.empty?
    end

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

    upgrade_formulae(formulae_to_install)

    check_installed_dependents

    Homebrew.messages.display_messages
  end
end
