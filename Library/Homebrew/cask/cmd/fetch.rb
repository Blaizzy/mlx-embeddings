# typed: false
# frozen_string_literal: true

module Cask
  class Cmd
    # Implementation of the `brew cask fetch` command.
    #
    # @api private
    class Fetch < AbstractCommand
      extend T::Sig

      sig { override.returns(T.nilable(T.any(Integer, Symbol))) }
      def self.min_named
        :cask
      end

      def self.parser
        super do
          switch "--force",
                 description: "Force redownloading even if files already exist in local cache."
        end
      end

      sig { returns(String) }
      def self.description
        "Downloads remote application files to local cache."
      end

      sig { void }
      def run
        require "cask/download"
        require "cask/installer"

        options = {
          force:      args.force?,
          quarantine: args.quarantine?,
        }.compact

        options[:quarantine] = true if options[:quarantine].nil?

        casks.each do |cask|
          puts Installer.caveats(cask)
          ohai "Downloading external files for Cask #{cask}"
          downloaded_path = Download.new(cask, **options).perform
          Verify.all(cask, downloaded_path)
          ohai "Success! Downloaded to -> #{downloaded_path}"
        end
      end
    end
  end
end
