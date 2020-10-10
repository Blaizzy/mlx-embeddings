# typed: false
# frozen_string_literal: true

require "delegate"

require "extend/hash_validator"
using HashValidator

module Cask
  class DSL
    # Class corresponding to the `conflicts_with` stanza.
    #
    # @api private
    class ConflictsWith < SimpleDelegator
      VALID_KEYS = [
        :formula,
        :cask,
        :macos,
        :arch,
        :x11,
        :java,
      ].freeze

      def initialize(**options)
        options.assert_valid_keys!(*VALID_KEYS)

        conflicts = options.transform_values { |v| Set.new(Array(v)) }
        conflicts.default = Set.new

        super(conflicts)
      end

      def to_json(generator)
        transform_values(&:to_a).to_json(generator)
      end
    end
  end
end
