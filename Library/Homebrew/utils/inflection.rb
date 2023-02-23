# typed: strict
# frozen_string_literal: true

module Utils
  # Inflection utility methods, as a lightweight alternative to `ActiveSupport::Inflector``.
  #
  # @api private
  module Inflection
    extend T::Sig

    # Combines `stem`` with the `singular`` or `plural` suffix based on `count`.
    sig { params(count: Integer, stem: String, plural: String, singular: String).returns(String) }
    def self.number(count, stem, plural = "s", singular = "")
      suffix = (count == 1) ? singular : plural
      "#{stem}#{suffix}"
    end
  end
end
