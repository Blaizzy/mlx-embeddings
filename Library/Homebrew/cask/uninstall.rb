# typed: false
# frozen_string_literal: true

module Cask
  # @api private
  class Uninstall
    def self.uninstall_casks(*casks, binaries: nil, force: false, verbose: false)
      require "cask/installer"

      casks.each do |cask|
        odebug "Uninstalling Cask #{cask}"

        raise CaskNotInstalledError, cask if !cask.installed? && !force

        Installer.new(cask, binaries: binaries, force: force, verbose: verbose).uninstall

        next if (versions = cask.versions).empty?

        puts <<~EOS
          #{cask} #{versions.to_sentence} #{(versions.count == 1) ? "is" : "are"} still installed.
          Remove #{(versions.count == 1) ? "it" : "them all"} with `brew uninstall --cask --force #{cask}`.
        EOS
      end
    end
  end
end
