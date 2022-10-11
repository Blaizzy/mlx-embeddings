# typed: false
# frozen_string_literal: true

require "set"

module Cask
  # Sorted set containing all cask artifacts.
  #
  # @api private
  class ArtifactSet < ::Set
    def each(&block)
      return enum_for(__method__) { size } unless block

      to_a.each(&block)
      self
    end

    def to_a
      super.sort
    end
  end
end
