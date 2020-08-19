# frozen_string_literal: true

require "cask/artifact/moved"

module Cask
  module Artifact
    # Artifact corresponding to the `prefpane` stanza.
    #
    # @api private
    class Prefpane < Moved
      def self.english_name
        "Preference Pane"
      end
    end
  end
end
