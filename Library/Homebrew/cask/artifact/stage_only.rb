# typed: true
# frozen_string_literal: true

require "cask/artifact/abstract_artifact"

module Cask
  module Artifact
    # Artifact corresponding to the `stage_only` stanza.
    #
    # @api private
    class StageOnly < AbstractArtifact
      extend T::Sig

      def self.from_args(cask, *args)
        raise CaskInvalidError.new(cask.token, "'stage_only' takes only a single argument: true") if args != [true]

        new(cask, true)
      end

      sig { returns(T::Array[T::Boolean]) }
      def to_a
        [true]
      end

      sig { returns(String) }
      def summarize
        "true"
      end
    end
  end
end
