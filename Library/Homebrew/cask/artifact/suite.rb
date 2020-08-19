# frozen_string_literal: true

require "cask/artifact/moved"

module Cask
  module Artifact
    # Artifact corresponding to the `suite` stanza.
    #
    # @api private
    class Suite < Moved
      def self.english_name
        "App Suite"
      end

      def self.dirmethod
        :appdir
      end
    end
  end
end
