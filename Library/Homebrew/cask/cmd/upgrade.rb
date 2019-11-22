# frozen_string_literal: true

require "cask/config"

module Cask
  class Cmd
    class Upgrade < AbstractCommand
      option "--greedy", :greedy, false
      option "--quiet",  :quiet, false
      option "--force", :force, false
      option "--skip-cask-deps", :skip_cask_deps, false
      option "--dry-run", :dry_run, false

      def initialize(*)
        super
        self.verbose = ($stdout.tty? || verbose?) && !quiet?
      end

      def run
        outdated_casks = casks(alternative: lambda {
          Caskroom.casks.select do |cask|
            cask.outdated?(greedy?)
          end
        }).select do |cask|
          raise CaskNotInstalledError, cask unless cask.installed? || force?

          cask.outdated?(true)
        end

        if outdated_casks.empty?
          oh1 "No Casks to upgrade"
          return
        end

        ohai "Casks with `auto_updates` or `version :latest` will not be upgraded" if args.empty? && !greedy?
        verb = dry_run? ? "Would upgrade" : "Upgrading"
        oh1 "#{verb} #{outdated_casks.count} #{"outdated package".pluralize(outdated_casks.count)}:"
        caught_exceptions = []

        upgradable_casks = outdated_casks.map { |c| [CaskLoader.load(c.installed_caskfile), c] }

        puts upgradable_casks
          .map { |(old_cask, new_cask)| "#{new_cask.full_name} #{old_cask.version} -> #{new_cask.version}" }
          .join(", ")
        return if dry_run?

        upgradable_casks.each do |(old_cask, new_cask)|
          upgrade_cask(old_cask, new_cask)
        rescue => e
          caught_exceptions << e
          next
        end

        return if caught_exceptions.empty?
        raise MultipleCaskErrors, caught_exceptions if caught_exceptions.count > 1
        raise caught_exceptions.first if caught_exceptions.count == 1
      end

      def upgrade_cask(old_cask, new_cask)
        odebug "Started upgrade process for Cask #{old_cask}"
        old_config = old_cask.config

        old_cask_installer =
          Installer.new(old_cask, binaries: binaries?,
                                  verbose:  verbose?,
                                  force:    force?,
                                  upgrade:  true)

        new_cask.config = Config.global.merge(old_config)

        new_cask_installer =
          Installer.new(new_cask, binaries:       binaries?,
                                  verbose:        verbose?,
                                  force:          force?,
                                  skip_cask_deps: skip_cask_deps?,
                                  require_sha:    require_sha?,
                                  upgrade:        true,
                                  quarantine:     quarantine?)

        started_upgrade = false
        new_artifacts_installed = false

        begin
          oh1 "Upgrading #{Formatter.identifier(old_cask)}"

          # Start new Cask's installation steps
          new_cask_installer.check_conflicts

          puts new_cask_installer.caveats if new_cask_installer.caveats

          new_cask_installer.fetch

          # Move the old Cask's artifacts back to staging
          old_cask_installer.start_upgrade
          # And flag it so in case of error
          started_upgrade = true

          # Install the new Cask
          new_cask_installer.stage

          new_cask_installer.install_artifacts
          new_artifacts_installed = true

          # If successful, wipe the old Cask from staging
          old_cask_installer.finalize_upgrade
        rescue => e
          new_cask_installer.uninstall_artifacts if new_artifacts_installed
          new_cask_installer.purge_versioned_files
          old_cask_installer.revert_upgrade if started_upgrade
          raise e
        end
      end

      def self.help
        "upgrades all outdated casks"
      end
    end
  end
end
