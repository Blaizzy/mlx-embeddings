# frozen_string_literal: true

module Cask
  class Cmd
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
        options = {
          binaries: binaries,
          force:    force,
          verbose:  verbose,
        }.compact

        casks.each do |cask|
          odebug "Uninstalling Cask #{cask}"

          raise CaskNotInstalledError, cask unless cask.installed? || force

          if cask.installed? && !cask.installed_caskfile.nil?
            # use the same cask file that was used for installation, if possible
            cask = CaskLoader.load(cask.installed_caskfile) if cask.installed_caskfile.exist?
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
