# frozen_string_literal: true

module Cask
  class Cmd
    # Implementation of the `brew cask install` command.
    #
    # @api private
    class Install < AbstractCommand
      def self.min_named
        :cask
      end

      def self.description
        "Installs the given <cask>."
      end

      def self.parser(&block)
        super do
          switch "--force",
                 description: "Force overwriting existing files."
          switch "--skip-cask-deps",
                 description: "Skip installing cask dependencies."

          instance_eval(&block) if block_given?
        end
      end

      def run
        require "cask/installer"

        options = {
          binaries:       args.binaries?,
          verbose:        args.verbose?,
          force:          args.force?,
          skip_cask_deps: args.skip_cask_deps?,
          require_sha:    args.require_sha?,
          quarantine:     args.quarantine?,
        }.compact

        options[:quarantine] = true if options[:quarantine].nil?

        odie "Installing casks is supported only on macOS" unless OS.mac?
        casks.each do |cask|
          Installer.new(cask, **options).install
        rescue CaskAlreadyInstalledError => e
          opoo e.message
        end
      end
    end
  end
end
