# typed: false
# frozen_string_literal: true

require "reinstall"
require "formula_installer"
require "development_tools"
require "messages"
require "cleanup"

module Homebrew
  # Helper functions for upgrading formulae.
  #
  # @api private
  module Upgrade
    module_function

    def upgrade_formulae(
      formulae_to_install,
      flags:,
      dry_run: false,
      installed_on_request: false,
      force_bottle: false,
      build_from_source_formulae: [],
      interactive: false,
      keep_tmp: false,
      force: false,
      debug: false,
      quiet: false,
      verbose: false
    )
      return if formulae_to_install.empty?

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

      formula_installers = formulae_to_install.map do |formula|
        Migrator.migrate_if_needed(formula, force: force, dry_run: dry_run)
        begin
          fi = create_formula_installer(
            formula,
            flags:                      flags,
            installed_on_request:       installed_on_request,
            force_bottle:               force_bottle,
            build_from_source_formulae: build_from_source_formulae,
            interactive:                interactive,
            keep_tmp:                   keep_tmp,
            force:                      force,
            debug:                      debug,
            quiet:                      quiet,
            verbose:                    verbose,
          )
          fi.fetch unless dry_run
          fi
        rescue UnsatisfiedRequirements, DownloadError => e
          ofail "#{formula}: #{e}"
          nil
        end
      end.compact

      formula_installers.each do |fi|
        upgrade_formula(fi, dry_run: dry_run, verbose: verbose)
        Cleanup.install_formula_clean!(fi.formula, dry_run: dry_run)
      end
    end

    def outdated_kegs(formula)
      [formula, *formula.old_installed_formulae].map(&:linked_keg)
                                                .select(&:directory?)
                                                .map { |k| Keg.new(k.resolved_path) }
    end

    def print_dry_run_dependencies(formula, fi_deps)
      return if fi_deps.empty?

      plural = "dependency".pluralize(fi_deps.count)
      ohai "Would upgrade #{fi_deps.count} #{plural} for #{formula.full_specified_name}:"
      formulae_upgrades = fi_deps.map(&:first).map(&:to_formula).map do |f|
        name = f.full_specified_name
        if f.optlinked?
          "#{name} #{Keg.new(f.opt_prefix).version} -> #{f.pkg_version}"
        else
          "#{name} #{f.pkg_version}"
        end
      end
      puts formulae_upgrades.join(", ")
    end

    def print_upgrade_message(formula, fi_options)
      version_upgrade = if formula.optlinked?
        "#{Keg.new(formula.opt_prefix).version} -> #{formula.pkg_version}"
      else
        "-> #{formula.pkg_version}"
      end
      oh1 <<~EOS
        Upgrading #{Formatter.identifier(formula.full_specified_name)}
          #{version_upgrade} #{fi_options.to_a.join(" ")}
      EOS
    end

    def create_formula_installer(
      formula,
      flags:,
      installed_on_request: false,
      force_bottle: false,
      build_from_source_formulae: [],
      interactive: false,
      keep_tmp: false,
      force: false,
      debug: false,
      quiet: false,
      verbose: false
    )
      if formula.opt_prefix.directory?
        keg = Keg.new(formula.opt_prefix.resolved_path)
        keg_had_linked_opt = true
        keg_was_linked = keg.linked?
      end

      if formula.opt_prefix.directory?
        keg = Keg.new(formula.opt_prefix.resolved_path)
        tab = Tab.for_keg(keg)
      end

      build_options = BuildOptions.new(Options.create(flags), formula.options)
      options = build_options.used_options
      options |= formula.build.used_options
      options &= formula.options

      FormulaInstaller.new(
        formula,
        **{
          options:                    options,
          link_keg:                   keg_had_linked_opt ? keg_was_linked : nil,
          installed_as_dependency:    tab&.installed_as_dependency,
          installed_on_request:       installed_on_request || tab&.installed_on_request,
          build_bottle:               tab&.built_bottle?,
          force_bottle:               force_bottle,
          build_from_source_formulae: build_from_source_formulae,
          interactive:                interactive,
          keep_tmp:                   keep_tmp,
          force:                      force,
          debug:                      debug,
          quiet:                      quiet,
          verbose:                    verbose,
        }.compact,
      )
    end
    private_class_method :create_formula_installer

    def upgrade_formula(formula_installer, dry_run: false, verbose: false)
      formula = formula_installer.formula

      kegs = outdated_kegs(formula)
      linked_kegs = kegs.select(&:linked?)

      if dry_run
        print_dry_run_dependencies(formula, formula_installer.compute_dependencies)
        return
      else
        print_upgrade_message(formula, formula_installer.options)
      end

      formula_installer.prelude

      # first we unlink the currently active keg for this formula otherwise it is
      # possible for the existing build to interfere with the build we are about to
      # do! Seriously, it happens!
      kegs.each(&:unlink)

      formula_installer.install
      formula_installer.finish
    rescue FormulaInstallationAlreadyAttemptedError
      # We already attempted to upgrade f as part of the dependency tree of
      # another formula. In that case, don't generate an error, just move on.
      nil
    rescue CannotInstallFormulaError => e
      ofail e
    rescue BuildError => e
      e.dump(verbose: verbose)
      puts
      Homebrew.failed = true
    ensure
      # restore previous installation state if build failed
      begin
        linked_kegs.each(&:link) unless formula.latest_version_installed?
      rescue
        nil
      end
    end
    private_class_method :upgrade_formula

    def check_broken_dependents(installed_formulae)
      CacheStoreDatabase.use(:linkage) do |db|
        installed_formulae.flat_map(&:runtime_installed_formula_dependents)
                          .uniq
                          .select do |f|
          keg = f.any_installed_keg
          next unless keg
          next unless keg.directory?

          LinkageChecker.new(keg, cache_db: db)
                        .broken_library_linkage?
        end.compact
      end
    end

    def check_installed_dependents(
      formulae,
      flags:,
      dry_run: false,
      installed_on_request: false,
      force_bottle: false,
      build_from_source_formulae: [],
      interactive: false,
      keep_tmp: false,
      force: false,
      debug: false,
      quiet: false,
      verbose: false
    )
      return if Homebrew::EnvConfig.no_installed_dependents_check?

      installed_formulae = dry_run ? formulae : FormulaInstaller.installed.to_a
      return if installed_formulae.empty?

      already_broken_dependents = check_broken_dependents(installed_formulae)

      outdated_dependents =
        installed_formulae.flat_map(&:runtime_installed_formula_dependents)
                          .uniq
                          .select(&:outdated?)
      return if outdated_dependents.blank? && already_broken_dependents.blank?

      outdated_dependents -= installed_formulae if dry_run

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
        ohai "No outdated dependents to upgrade!" unless dry_run
      else
        plural = "dependent".pluralize(upgradeable_dependents.count)
        verb = dry_run ? "Would upgrade" : "Upgrading"
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

      unless dry_run
        upgrade_formulae(
          upgradeable_dependents,
          flags:                      flags,
          installed_on_request:       installed_on_request,
          force_bottle:               force_bottle,
          build_from_source_formulae: build_from_source_formulae,
          interactive:                interactive,
          keep_tmp:                   keep_tmp,
          force:                      force,
          debug:                      debug,
          quiet:                      quiet,
          verbose:                    verbose,
        )
      end

      # Update installed formulae after upgrading
      installed_formulae = FormulaInstaller.installed.to_a

      # Assess the dependents tree again now we've upgraded.
      oh1 "Checking for dependents of upgraded formulae..." unless dry_run
      broken_dependents = check_broken_dependents(installed_formulae)
      if broken_dependents.blank?
        if dry_run
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

      return if dry_run

      reinstallable_broken_dependents.each do |formula|
        Homebrew.reinstall_formula(
          formula,
          flags:                      flags,
          force_bottle:               force_bottle,
          build_from_source_formulae: build_from_source_formulae + [formula.full_name],
          interactive:                interactive,
          keep_tmp:                   keep_tmp,
          force:                      force,
          debug:                      debug,
          quiet:                      quiet,
          verbose:                    verbose,
        )
      rescue FormulaInstallationAlreadyAttemptedError
        # We already attempted to reinstall f as part of the dependency tree of
        # another formula. In that case, don't generate an error, just move on.
        nil
      rescue CannotInstallFormulaError, DownloadError => e
        ofail e
      rescue BuildError => e
        e.dump(verbose: verbose)
        puts
        Homebrew.failed = true
      end
    end

    def depends_on(a, b)
      if a.any_installed_keg
         &.runtime_dependencies
         &.any? { |d| d["full_name"] == b.full_name }
        1
      else
        a <=> b
      end
    end
    private_class_method :depends_on
  end
end
