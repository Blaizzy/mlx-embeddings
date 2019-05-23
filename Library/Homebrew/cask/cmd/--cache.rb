# frozen_string_literal: true

require "cask/download"

module Cask
  class Cmd
    class Cache < AbstractCommand
      def self.command_name
        "--cache"
      end

      def initialize(*)
        super
        raise CaskUnspecifiedError if args.empty?
      end

      def run
        casks.each do |cask|
          puts Download.new(cask).downloader.cached_location
        end
      end

      def self.help
        "display the file used to cache the Cask"
      end
    end
  end
end
