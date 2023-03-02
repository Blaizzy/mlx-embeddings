# typed: true
# frozen_string_literal: true

class Array
  # Converts the array to a comma-separated sentence where the last element is
  # joined by the connector word.
  #
  # You can pass the following kwargs to change the default behavior:
  #
  # * <tt>:words_connector</tt> - The sign or word used to join all but the last
  #   element in arrays with three or more elements (default: ", ").
  # * <tt>:last_word_connector</tt> - The sign or word used to join the last element
  #   in arrays with three or more elements (default: ", and ").
  # * <tt>:two_words_connector</tt> - The sign or word used to join the elements
  #   in arrays with two elements (default: " and ").
  #
  # ==== Examples
  #
  #   [].to_sentence                      # => ""
  #   ['one'].to_sentence                 # => "one"
  #   ['one', 'two'].to_sentence          # => "one and two"
  #   ['one', 'two', 'three'].to_sentence # => "one, two, and three"
  #   ['one', 'two'].to_sentence(two_words_connector: '-')
  #   # => "one-two"
  #
  #   ['one', 'two', 'three'].to_sentence(words_connector: ' or ', last_word_connector: ' or at least ')
  #   # => "one or two or at least three"
  # @see https://github.com/rails/rails/blob/v7.0.4.2/activesupport/lib/active_support/core_ext/array/conversions.rb#L8-L84
  #   ActiveSupport Array#to_sentence monkey-patch
  def to_sentence(words_connector: ", ", two_words_connector: " and ", last_word_connector: ", and ")
    case length
    when 0
      +""
    when 1
      # This is not typesafe, if the array contains a BasicObject
      +T.unsafe(self[0]).to_s
    when 2
      +"#{self[0]}#{two_words_connector}#{self[1]}"
    else
      +"#{T.must(self[0...-1]).join(words_connector)}#{last_word_connector}#{self[-1]}"
    end
  end
end
