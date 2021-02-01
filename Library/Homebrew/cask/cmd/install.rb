# typed: false
# frozen_string_literal: true

module Cask
  class Cmd
    # Cask implementation of the `brew install` command.
    #
    # @api private
    class Install < AbstractCommand
      extend T::Sig

      OPTIONS = [
        [:switch, "--skip-cask-deps", {
          description: "Skip installing cask dependencies.",
        }],
      ].freeze

      def self.parser(&block)
        super do
          switch "--force",
                 description: "Force overwriting existing files."

          OPTIONS.each do |option|
            send(*option)
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
          skip_cask_deps: args.skip_cask_deps?,
          require_sha:    args.require_sha?,
          quarantine:     args.quarantine?,
        )
      end

      def self.install_casks(
        *casks,
        verbose: nil,
        force: nil,
        binaries: nil,
        skip_cask_deps: nil,
        require_sha: nil,
        quarantine: nil
      )
        odie "Installing casks is supported only on macOS" unless OS.mac?

        options = {
          verbose:        verbose,
          force:          force,
          binaries:       binaries,
          skip_cask_deps: skip_cask_deps,
          require_sha:    require_sha,
          quarantine:     quarantine,
        }.compact

        options[:quarantine] = true if options[:quarantine].nil?

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
