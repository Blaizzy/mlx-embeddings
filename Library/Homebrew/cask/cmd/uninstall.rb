# frozen_string_literal: true

module Cask
  class Cmd
    # Implementation of the `brew cask uninstall` command.
    #
    # @api private
    class Uninstall < AbstractCommand
      def self.min_named
        :cask
      end

      def self.description
        "Uninstalls the given <cask>."
      end

      def self.parser
        super do
          switch "--force",
                 description: "Uninstall even if the <cask> is not installed, overwrite " \
                              "existing files and ignore errors when removing files."
        end
      end

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

          if cask.installed?
            if (installed_caskfile = cask.installed_caskfile) && installed_caskfile.exist?
              # Use the same cask file that was used for installation, if possible.
              cask = CaskLoader.load(installed_caskfile)
            end
          else
            raise CaskNotInstalledError, cask unless force
          end

          Installer.new(cask, **options).uninstall

          next if (versions = cask.versions).empty?

          puts <<~EOS
            #{cask} #{versions.to_sentence} #{"is".pluralize(versions.count)} still installed.
            Remove #{(versions.count == 1) ? "it" : "them all"} with `brew cask uninstall --force #{cask}`.
          EOS
        end
      end
    end
  end
end
