# typed: false
# frozen_string_literal: true

module Cask
  class Cmd
    # Cask implementation of the `brew uninstall` command.
    #
    # @api private
    class Uninstall < AbstractCommand
      extend T::Sig

      def self.parser
        super do
          switch "--force",
                 description: "Uninstall even if the <cask> is not installed, overwrite " \
                              "existing files and ignore errors when removing files."
        end
      end

      sig { void }
      def run
        self.class.uninstall_casks(
          *casks,
          binaries: args.binaries?,
          verbose:  args.verbose?,
          force:    args.force?,
        )
      end

      def self.uninstall_casks(*casks, binaries: nil, force: false, verbose: false)
        require "cask/installer"

        options = {
          binaries: binaries,
          force:    force,
          verbose:  verbose,
        }.compact

        casks.each do |cask|
          odebug "Uninstalling Cask #{cask}"

          raise CaskNotInstalledError, cask if !cask.installed? && !force

          Installer.new(cask, **options).uninstall

          next if (versions = cask.versions).empty?

          puts <<~EOS
            #{cask} #{versions.to_sentence} #{"is".pluralize(versions.count)} still installed.
            Remove #{(versions.count == 1) ? "it" : "them all"} with `brew uninstall --cask --force #{cask}`.
          EOS
        end
      end
    end
  end
end
