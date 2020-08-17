# frozen_string_literal: true

require "cask/download"

module Cask
  class Cmd
    class Fetch < AbstractCommand
      def self.min_named
        :cask
      end

      def self.parser
        super do
          switch "--force",
                 description: "Force redownloading even if files already exist in local cache."
        end
      end

      def self.description
        "Downloads remote application files to local cache."
      end

      def run
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
