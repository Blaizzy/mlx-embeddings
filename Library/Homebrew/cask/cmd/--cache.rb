# frozen_string_literal: true

require "cask/download"

module Cask
  class Cmd
    class Cache < AbstractCommand
      def self.min_named
        :cask
      end

      def self.description
        "Display the file used to cache a <cask>."
      end

      def self.command_name
        "--cache"
      end

      def run
        # TODO: enable for next major/minor release
        # odeprecated "brew cask --cache", "brew --cache --cask"

        casks.each do |cask|
          puts self.class.cached_location(cask)
        end
      end

      def self.cached_location(cask)
        Download.new(cask).downloader.cached_location
      end
    end
  end
end
