# typed: false
# frozen_string_literal: true

require "cask_dependent"

module Cask
  class Cmd
    # Cask implementation of the `brew install` command.
    #
    # @api private
    class Install < AbstractCommand
      extend T::Sig

      OPTIONS = [
        [:switch, "--adopt", {
          description: "Adopt existing artifacts in the destination that are identical to those being installed. " \
                       "Cannot be combined with --force.",
        }],
        [:switch, "--skip-cask-deps", {
          description: "Skip installing cask dependencies.",
        }],
        [:switch, "--zap", {
          description: "For use with `brew reinstall --cask`. Remove all files associated with a cask. " \
                       "*May remove files which are shared between applications.*",
        }],
      ].freeze

      def self.parser(&block)
        super do
          switch "--force",
                 description: "Force overwriting existing files."

          OPTIONS.map(&:dup).each do |option|
            kwargs = option.pop
            send(*option, **kwargs)
          end

          instance_eval(&block) if block
        end
      end

      sig { void }
      def run
        self.class.install_casks(
          *casks,
          binaries:       args.binaries?,
          verbose:        args.verbose?,
          force:          args.force?,
          adopt:          args.adopt?,
          skip_cask_deps: args.skip_cask_deps?,
          require_sha:    args.require_sha?,
          quarantine:     args.quarantine?,
          quiet:          args.quiet?,
          zap:            args.zap?,
        )
      end

      def self.install_casks(
        *casks,
        verbose: nil,
        force: nil,
        adopt: nil,
        binaries: nil,
        skip_cask_deps: nil,
        require_sha: nil,
        quarantine: nil,
        quiet: nil,
        zap: nil,
        dry_run: nil
      )
        # TODO: Refactor and move to extend/os
        odie "Installing casks is supported only on macOS" unless OS.mac? # rubocop:disable Homebrew/MoveToExtendOS

        options = {
          verbose:        verbose,
          force:          force,
          adopt:          adopt,
          binaries:       binaries,
          skip_cask_deps: skip_cask_deps,
          require_sha:    require_sha,
          quarantine:     quarantine,
          quiet:          quiet,
          zap:            zap,
        }.compact

        options[:quarantine] = true if options[:quarantine].nil?

        if dry_run
          if (casks_to_install = casks.reject(&:installed?).presence)
            plural = "cask".pluralize(casks_to_install.count)
            ohai "Would install #{casks_to_install.count} #{plural}:"
            puts casks_to_install.map(&:full_name).join(" ")
          end
          casks.each do |cask|
            dep_names = CaskDependent.new(cask)
                                     .runtime_dependencies
                                     .reject(&:installed?)
                                     .map(&:to_formula)
                                     .map(&:name)
            next if dep_names.blank?

            plural = "dependency".pluralize(dep_names.count)
            ohai "Would install #{dep_names.count} #{plural} for #{cask.full_name}:"
            puts dep_names.join(" ")
          end
          return
        end

        require "cask/installer"

        casks.each do |cask|
          Installer.new(cask, **options).install
        rescue CaskAlreadyInstalledError => e
          opoo e.message
        end
      end
    end
  end
end
