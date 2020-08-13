# frozen_string_literal: true

module Cask
  class Cmd
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
        casks.each do |cask|
          odebug "Zapping Cask #{cask}"

          raise CaskNotInstalledError, cask unless cask.installed? || args.force?

          Installer.new(cask, verbose: args.verbose?, force: args.force?).zap
        end
      end
    end
  end
end
