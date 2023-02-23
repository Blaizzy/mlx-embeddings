# typed: true
# frozen_string_literal: true

require "set"

module Cask
  # Sorted set containing all cask artifacts.
  #
  # @api private
  class ArtifactSet < ::Set
    def each(&block)
      # TODO: This is a false positive: https://github.com/rubocop/rubocop/issues/11591
      # rubocop:disable Lint/ToEnumArguments
      return enum_for(T.must(__method__)) { size } unless block
      # rubocop:enable Lint/ToEnumArguments

      to_a.each(&block)
      self
    end

    def to_a
      super.sort
    end
  end
end
