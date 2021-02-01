# typed: true
# frozen_string_literal: true

module Cask
  class Cmd
    # Cask implementation of the `brew reinstall` command.
    #
    # @api private
    class Reinstall < Install
      extend T::Sig

      sig { void }
      def run
        self.class.reinstall_casks(
          *casks,
          binaries:       args.binaries?,
          verbose:        args.verbose?,
          force:          args.force?,
          skip_cask_deps: args.skip_cask_deps?,
          require_sha:    args.require_sha?,
          quarantine:     args.quarantine?,
        )
      end

      def self.reinstall_casks(
        *casks,
        verbose: nil,
        force: nil,
        skip_cask_deps: nil,
        binaries: nil,
        require_sha: nil,
        quarantine: nil
      )
        require "cask/installer"

        options = {
          binaries:       binaries,
          verbose:        verbose,
          force:          force,
          skip_cask_deps: skip_cask_deps,
          require_sha:    require_sha,
          quarantine:     quarantine,
        }.compact

        options[:quarantine] = true if options[:quarantine].nil?

        casks.each do |cask|
          Installer.new(cask, **options).reinstall
        end
      end
    end
  end
end
