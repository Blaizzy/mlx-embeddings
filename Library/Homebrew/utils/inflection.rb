# typed: strict
# frozen_string_literal: true

module Utils
  # Inflection utility methods, as a lightweight alternative to `ActiveSupport::Inflector``.
  #
  # @api private
  module Inflection
    extend T::Sig
    # Removes the rightmost segment from the constant expression in the string.
    #
    #   deconstantize('Net::HTTP')   # => "Net"
    #   deconstantize('::Net::HTTP') # => "::Net"
    #   deconstantize('String')      # => ""
    #   deconstantize('::String')    # => ""
    #   deconstantize('')            # => ""
    #
    # See also #demodulize.
    # @see https://github.com/rails/rails/blob/b0dd7c7/activesupport/lib/active_support/inflector/methods.rb#L247-L258
    #   `ActiveSupport::Inflector.deconstantize`
    sig { params(path: String).returns(String) }
    def self.deconstantize(path)
      T.must(path[0, path.rindex("::") || 0]) # implementation based on the one in facets' Module#spacename
    end

    # Removes the module part from the expression in the string.
    #
    #   demodulize('ActiveSupport::Inflector::Inflections') # => "Inflections"
    #   demodulize('Inflections')                           # => "Inflections"
    #   demodulize('::Inflections')                         # => "Inflections"
    #   demodulize('')                                      # => ""
    #
    # See also #deconstantize.
    # @see https://github.com/rails/rails/blob/b0dd7c7/activesupport/lib/active_support/inflector/methods.rb#L230-L245
    #   `ActiveSupport::Inflector.demodulize`
    sig { params(path: String).returns(String) }
    def self.demodulize(path)
      if (i = path.rindex("::"))
        T.must(path[(i + 2)..])
      else
        path
      end
    end

    # Combines `stem` with the `singular` or `plural` suffix based on `count`.
    sig { params(stem: String, count: Integer, plural: String, singular: String).returns(String) }
    def self.pluralize(stem, count, plural: "s", singular: "")
      suffix = (count == 1) ? singular : plural
      "#{stem}#{suffix}"
    end
  end
end
