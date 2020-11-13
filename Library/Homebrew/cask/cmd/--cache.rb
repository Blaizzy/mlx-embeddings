# typed: true
# frozen_string_literal: true

module Cask
  class Cmd
    # Implementation of the `brew cask --cache` command.
    #
    # @api private
    class Cache < AbstractCommand
      extend T::Sig

      sig { override.returns(T.nilable(T.any(Integer, Symbol))) }
      def self.min_named
        :cask
      end

      sig { returns(String) }
      def self.description
        "Display the file used to cache a <cask>."
      end

      sig { returns(String) }
      def self.command_name
        "--cache"
      end

      sig { void }
      def run
        casks.each do |cask|
          puts self.class.cached_location(cask)
        end
      end

      def self.cached_location(cask)
        require "cask/download"

        Download.new(cask).downloader.cached_location
      end
    end
  end
end
