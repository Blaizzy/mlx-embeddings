# typed: false
# frozen_string_literal: true

module Cask
  class Cmd
    # Cask implementation for the `brew uninstall` command.
    #
    # @api private
    class Zap < AbstractCommand
      extend T::Sig

      def self.parser
        super do
          switch "--force",
                 description: "Ignore errors when removing files."
        end
      end

      sig { void }
      def run
        self.class.zap_casks(*casks, verbose: args.verbose?, force: args.force?)
      end

      sig { params(casks: Cask, force: T.nilable(T::Boolean), verbose: T.nilable(T::Boolean)).void }
      def self.zap_casks(
        *casks,
        force: nil,
        verbose: nil
      )
        require "cask/installer"

        casks.each do |cask|
          odebug "Zapping Cask #{cask}"

          raise CaskNotInstalledError, cask if !cask.installed? && !force

          Installer.new(cask, verbose: verbose, force: force).zap
        end
      end
    end
  end
end
