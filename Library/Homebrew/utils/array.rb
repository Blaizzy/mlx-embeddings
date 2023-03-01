# typed: strict
# frozen_string_literal: true

module Utils
  extend T::Sig

  # Converts the array to a comma-separated sentence where the last element is
  # joined by the connector word.
  #
  # You can pass the following options to change the default behavior. If you
  # pass an option key that doesn't exist in the list below, it will raise an
  # <tt>ArgumentError</tt>.
  #
  # ==== Options
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
  #
  #   ['one', 'two'].to_sentence(two_words_connector: '-')
  #   # => "one-two"
  #
  #   ['one', 'two', 'three'].to_sentence(words_connector: ' or ', last_word_connector: ' or at least ')
  #   # => "one or two or at least three"
  # @see https://github.com/rails/rails/blob/v6.1.7.2/activesupport/lib/active_support/core_ext/array/conversions.rb#L10-L85
  #   ActiveSupport implementation
  sig {
    params(array: T::Array[String], words_connector: String, two_words_connector: String, last_word_connector: String)
      .returns(String)
  }
  def self.to_sentence(array, words_connector: ", ", two_words_connector: " and ", last_word_connector: ", and ")
    case array.length
    when 0
      +""
    when 1
      +(array[0]).to_s
    when 2
      +"#{array[0]}#{two_words_connector}#{array[1]}"
    else
      +"#{T.must(array[0...-1]).join(words_connector)}#{last_word_connector}#{array[-1]}"
    end
  end
end
