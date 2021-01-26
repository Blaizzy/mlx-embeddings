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

    def upgrade_formulae(formulae_to_install, args:)
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
        Migrator.migrate_if_needed(f, force: args.force?)
        begin
          upgrade_formula(f, args: args)
          Cleanup.install_formula_clean!(f)
        rescue UnsatisfiedRequirements => e
          ofail "#{f}: #{e}"
        end
      end
    end

    def upgrade_formula(f, args:)
      return if args.dry_run?

      if f.opt_prefix.directory?
        keg = Keg.new(f.opt_prefix.resolved_path)
        keg_had_linked_opt = true
        keg_was_linked = keg.linked?
      end

      formulae_maybe_with_kegs = [f] + f.old_installed_formulae
      outdated_kegs = formulae_maybe_with_kegs.map(&:linked_keg)
                                              .select(&:directory?)
                                              .map { |k| Keg.new(k.resolved_path) }
      linked_kegs = outdated_kegs.select(&:linked?)

      if f.opt_prefix.directory?
        keg = Keg.new(f.opt_prefix.resolved_path)
        tab = Tab.for_keg(keg)
      end

      build_options = BuildOptions.new(Options.create(args.flags_only), f.options)
      options = build_options.used_options
      options |= f.build.used_options
      options &= f.options

      fi = FormulaInstaller.new(
        f,
        **{
          options:                    options,
          link_keg:                   keg_had_linked_opt ? keg_was_linked : nil,
          installed_as_dependency:    tab&.installed_as_dependency,
          installed_on_request:       args.named.present? || tab&.installed_on_request,
          build_bottle:               args.build_bottle? || tab&.built_bottle?,
          force_bottle:               args.force_bottle?,
          build_from_source_formulae: args.build_from_source_formulae,
          keep_tmp:                   args.keep_tmp?,
          force:                      args.force?,
          debug:                      args.debug?,
          quiet:                      args.quiet?,
          verbose:                    args.verbose?,
        }.compact,
      )

      upgrade_version = if f.optlinked?
        "#{Keg.new(f.opt_prefix).version} -> #{f.pkg_version}"
      else
        "-> #{f.pkg_version}"
      end
      oh1 "Upgrading #{Formatter.identifier(f.full_specified_name)} #{upgrade_version} #{fi.options.to_a.join(" ")}"

      fi.prelude
      fi.fetch

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
    rescue CannotInstallFormulaError, DownloadError => e
      ofail e
    rescue BuildError => e
      e.dump(verbose: args.verbose?)
      puts
      Homebrew.failed = true
    ensure
      # restore previous installation state if build failed
      begin
        linked_kegs.each(&:link) unless f.latest_version_installed?
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

    def check_installed_dependents(formulae, args:)
      return if Homebrew::EnvConfig.no_installed_dependents_check?

      installed_formulae = args.dry_run? ? formulae : FormulaInstaller.installed.to_a
      return if installed_formulae.empty?

      already_broken_dependents = check_broken_dependents(installed_formulae)

      outdated_dependents =
        installed_formulae.flat_map(&:runtime_installed_formula_dependents)
                          .uniq
                          .select(&:outdated?)
      return if outdated_dependents.blank? && already_broken_dependents.blank?

      outdated_dependents -= installed_formulae if args.dry_run?

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

      upgrade_formulae(upgradeable_dependents, args: args)

      # Update installed formulae after upgrading
      installed_formulae = FormulaInstaller.installed.to_a

      # Assess the dependents tree again now we've upgraded.
      oh1 "Checking for dependents of upgraded formulae..." unless args.dry_run?
      broken_dependents = check_broken_dependents(installed_formulae)
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

      return if args.dry_run?

      reinstallable_broken_dependents.each do |f|
        Homebrew.reinstall_formula(f, build_from_source: true, args: args)
      rescue FormulaInstallationAlreadyAttemptedError
        # We already attempted to reinstall f as part of the dependency tree of
        # another formula. In that case, don't generate an error, just move on.
        nil
      rescue CannotInstallFormulaError, DownloadError => e
        ofail e
      rescue BuildError => e
        e.dump(verbose: args.verbose?)
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
