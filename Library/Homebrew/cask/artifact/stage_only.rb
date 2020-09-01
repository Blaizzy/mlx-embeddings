# frozen_string_literal: true

require "cask/artifact/abstract_artifact"

module Cask
  module Artifact
    # Artifact corresponding to the `stage_only` stanza.
    #
    # @api private
    class StageOnly < AbstractArtifact
      def self.from_args(cask, *args)
        raise CaskInvalidError.new(cask.token, "'stage_only' takes only a single argument: true") if args != [true]

        new(cask)
      end

      def to_a
        [true]
      end

      def summarize
        "true"
      end
    end
  end
end
