# typed: false
# frozen_string_literal: true

module Cask
  class Cmd
    # Cask implementation of the `brew fetch` command.
    #
    # @api private
    class Fetch < AbstractCommand
      extend T::Sig

      def self.parser
        super do
          switch "--force",
                 description: "Force redownloading even if files already exist in local cache."
        end
      end

      sig { void }
      def run
        require "cask/download"
        require "cask/installer"

        options = {
          quarantine: args.quarantine?,
        }.compact

        options[:quarantine] = true if options[:quarantine].nil?

        casks.each do |cask|
          puts Installer.caveats(cask)
          ohai "Downloading external files for Cask #{cask}"
          download = Download.new(cask, **options)
          download.clear_cache if args.force?
          downloaded_path = download.fetch
          ohai "Success! Downloaded to: #{downloaded_path}"
        end
      end
    end
  end
end
