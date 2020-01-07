# frozen_string_literal: true

require "install"
require "reinstall"
require "formula_installer"
require "development_tools"
require "messages"
require "cleanup"
require "cli/parser"

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

    if Homebrew.args.named.blank?
      outdated = Formula.installed.select do |f|
        f.outdated?(fetch_head: args.fetch_HEAD?)
      end

      exit 0 if outdated.empty?
    else
      outdated = Homebrew.args.resolved_formulae.select do |f|
        f.outdated?(fetch_head: args.fetch_HEAD?)
      end

      (Homebrew.args.resolved_formulae - outdated).each do |f|
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

    check_dependents(formulae_to_install)

    Homebrew.messages.display_messages
  end

  def upgrade_formulae(formulae_to_install)
    return if formulae_to_install.empty?
    return if args.dry_run?

    # Sort keg-only before non-keg-only formulae to avoid any needless conflicts
    # with outdated, non-keg-only versions of formulae being upgraded.
    formulae_to_install.sort! do |a, b|
      if !a.keg_only? && b.keg_only?
        1
      elsif a.keg_only? && !b.keg_only?
        -1
      else
        0
      end
    end

    formulae_to_install.each do |f|
      Migrator.migrate_if_needed(f)
      begin
        upgrade_formula(f)
        Cleanup.install_formula_clean!(f)
      rescue UnsatisfiedRequirements => e
        Homebrew.failed = true
        onoe "#{f}: #{e}"
      end
    end
  end

  def upgrade_formula(f)
    return if args.dry_run?

    if f.opt_prefix.directory?
      keg = Keg.new(f.opt_prefix.resolved_path)
      keg_had_linked_opt = true
      keg_was_linked = keg.linked?
    end

    formulae_maybe_with_kegs = [f] + f.old_installed_formulae
    outdated_kegs = formulae_maybe_with_kegs
                    .map(&:linked_keg)
                    .select(&:directory?)
                    .map { |k| Keg.new(k.resolved_path) }
    linked_kegs = outdated_kegs.select(&:linked?)

    if f.opt_prefix.directory?
      keg = Keg.new(f.opt_prefix.resolved_path)
      tab = Tab.for_keg(keg)
    end

    build_options = BuildOptions.new(Options.create(Homebrew.args.flags_only), f.options)
    options = build_options.used_options
    options |= f.build.used_options
    options &= f.options

    fi = FormulaInstaller.new(f)
    fi.options = options
    fi.build_bottle = args.build_bottle?
    fi.installed_on_request = Homebrew.args.named.present?
    fi.link_keg           ||= keg_was_linked if keg_had_linked_opt
    if tab
      fi.build_bottle          ||= tab.built_bottle?
      fi.installed_as_dependency = tab.installed_as_dependency
      fi.installed_on_request  ||= tab.installed_on_request
    end
    fi.prelude

    oh1 "Upgrading #{Formatter.identifier(f.full_specified_name)} #{fi.options.to_a.join " "}"

    # first we unlink the currently active keg for this formula otherwise it is
    # possible for the existing build to interfere with the build we are about to
    # do! Seriously, it happens!
    outdated_kegs.each(&:unlink)

    fi.install
    fi.finish
  rescue FormulaInstallationAlreadyAttemptedError
    # We already attempted to upgrade f as part of the dependency tree of
    # another formula. In that case, don't generate an error, just move on.
    nil
  rescue CannotInstallFormulaError => e
    ofail e
  rescue BuildError => e
    e.dump
    puts
    Homebrew.failed = true
  rescue DownloadError => e
    ofail e
  ensure
    # restore previous installation state if build failed
    begin
      linked_kegs.each(&:link) unless f.installed?
    rescue
      nil
    end
  end

  # @private
  def depends_on(a, b)
    if a.opt_or_installed_prefix_keg
        .runtime_dependencies
        .any? { |d| d["full_name"] == b.full_name }
      1
    else
      a <=> b
    end
  end

  def check_dependents(formulae_to_install)
    return if formulae_to_install.empty?

    oh1 "Checking for dependents of upgraded formulae..." unless args.dry_run?
    outdated_dependents =
      formulae_to_install.flat_map(&:runtime_installed_formula_dependents)
                         .select(&:outdated?)
    if outdated_dependents.blank?
      ohai "No dependents found!" unless args.dry_run?
      return
    end
    outdated_dependents -= formulae_to_install if args.dry_run?

    upgradeable_dependents =
      outdated_dependents.reject(&:pinned?)
                         .sort { |a, b| depends_on(a, b) }
    pinned_dependents =
      outdated_dependents.select(&:pinned?)
                         .sort { |a, b| depends_on(a, b) }

    if pinned_dependents.present?
      plural = "dependent".pluralize(pinned_dependents.count)
      ohai "Not upgrading #{pinned_dependents.count} pinned #{plural}:"
      puts(pinned_dependents.map do |f|
        "#{f.full_specified_name} #{f.pkg_version}"
      end.join(", "))
    end

    # Print the upgradable dependents.
    if upgradeable_dependents.blank?
      ohai "No outdated dependents to upgrade!" unless args.dry_run?
    else
      plural = "dependent".pluralize(upgradeable_dependents.count)
      verb = args.dry_run? ? "Would upgrade" : "Upgrading"
      ohai "#{verb} #{upgradeable_dependents.count} #{plural}:"
      formulae_upgrades = upgradeable_dependents.map do |f|
        name = f.full_specified_name
        if f.optlinked?
          "#{name} #{Keg.new(f.opt_prefix).version} -> #{f.pkg_version}"
        else
          "#{name} #{f.pkg_version}"
        end
      end
      puts formulae_upgrades.join(", ")
    end

    upgrade_formulae(upgradeable_dependents)

    # Assess the dependents tree again now we've upgraded.
    oh1 "Checking for dependents of upgraded formulae..." unless args.dry_run?
    broken_dependents = CacheStoreDatabase.use(:linkage) do |db|
      formulae_to_install.flat_map(&:runtime_installed_formula_dependents)
                         .select do |f|
        keg = f.opt_or_installed_prefix_keg
        next unless keg

        LinkageChecker.new(keg, cache_db: db)
                      .broken_library_linkage?
      end.compact
    end
    if broken_dependents.blank?
      if args.dry_run?
        ohai "No currently broken dependents found!"
        opoo "If they are broken by the upgrade they will also be upgraded or reinstalled."
      else
        ohai "No broken dependents found!"
      end
      return
    end

    reinstallable_broken_dependents =
      broken_dependents.reject(&:outdated?)
                       .reject(&:pinned?)
                       .sort { |a, b| depends_on(a, b) }
    outdated_pinned_broken_dependents =
      broken_dependents.select(&:outdated?)
                       .select(&:pinned?)
                       .sort { |a, b| depends_on(a, b) }

    # Print the pinned dependents.
    if outdated_pinned_broken_dependents.present?
      count = outdated_pinned_broken_dependents.count
      plural = "dependent".pluralize(outdated_pinned_broken_dependents.count)
      onoe "Not reinstalling #{count} broken and outdated, but pinned #{plural}:"
      $stderr.puts(outdated_pinned_broken_dependents.map do |f|
        "#{f.full_specified_name} #{f.pkg_version}"
      end.join(", "))
    end

    # Print the broken dependents.
    if reinstallable_broken_dependents.blank?
      ohai "No broken dependents to reinstall!"
    else
      count = reinstallable_broken_dependents.count
      plural = "dependent".pluralize(reinstallable_broken_dependents.count)
      ohai "Reinstalling #{count} broken #{plural} from source:"
      puts reinstallable_broken_dependents.map(&:full_specified_name)
                                          .join(", ")
    end

    reinstallable_broken_dependents.each do |f|
      reinstall_formula(f, build_from_source: true)
    rescue FormulaInstallationAlreadyAttemptedError
      # We already attempted to reinstall f as part of the dependency tree of
      # another formula. In that case, don't generate an error, just move on.
      nil
    rescue CannotInstallFormulaError => e
      ofail e
    rescue BuildError => e
      e.dump
      puts
      Homebrew.failed = true
    rescue DownloadError => e
      ofail e
    end
  end
end
