# frozen_string_literal: true

require "extend/hash_validator"
using HashValidator

module Cask
  class DSL
    class ConflictsWith < DelegateClass(Hash)
      VALID_KEYS = [
        :formula,
        :cask,
        :macos,
        :arch,
        :x11,
        :java,
      ].freeze

      def initialize(**pairs)
        pairs.assert_valid_keys!(*VALID_KEYS)

        super(pairs.transform_values { |v| Set.new([*v]) })

        self.default = Set.new
      end

      def to_json(generator)
        transform_values(&:to_a).to_json(generator)
      end
    end
  end
end
