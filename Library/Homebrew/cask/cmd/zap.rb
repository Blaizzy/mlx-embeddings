# frozen_string_literal: true

module Cask
  class Cmd
    # Implementation of the `brew cask zap` command.
    #
    # @api private
    class Zap < AbstractCommand
      def self.min_named
        :cask
      end

      def self.description
        <<~EOS
          Zaps all files associated with the given <cask>. Implicitly also performs all actions associated with `uninstall`.

          *May remove files which are shared between applications.*
        EOS
      end

      def self.parser
        super do
          switch "--force",
                 description: "Ignore errors when removing files."
        end
      end

      def run
        require "cask/installer"

        casks.each do |cask|
          odebug "Zapping Cask #{cask}"

          if cask.installed?
            if (installed_caskfile = cask.installed_caskfile) && installed_caskfile.exist?
              # Use the same cask file that was used for installation, if possible.
              cask = CaskLoader.load(installed_caskfile)
            end
          else
            raise CaskNotInstalledError, cask unless args.force?
          end

          Installer.new(cask, verbose: args.verbose?, force: args.force?).zap
        end
      end
    end
  end
end
