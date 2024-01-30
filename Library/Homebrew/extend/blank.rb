# typed: strict
# frozen_string_literal: true

class Object
  # An object is blank if it's false, empty, or a whitespace string.
  # For example, +nil+, '', '   ', [], {}, and +false+ are all blank.
  #
  # This simplifies
  #
  #   !address || address.empty?
  #
  # to
  #
  #   address.blank?
  sig { returns(T::Boolean) }
  def blank?
    respond_to?(:empty?) ? !!T.unsafe(self).empty? : false
  end

  # An object is present if it's not blank.
  sig { returns(T::Boolean) }
  def present? = !blank?

  # Returns the receiver if it's present otherwise returns +nil+.
  # <tt>object.presence</tt> is equivalent to
  #
  #    object.present? ? object : nil
  #
  # For example, something like
  #
  #   state   = params[:state]   if params[:state].present?
  #   country = params[:country] if params[:country].present?
  #   region  = state || country || 'US'
  #
  # becomes
  #
  #   region = params[:state].presence || params[:country].presence || 'US'
  sig { returns(T.nilable(T.self_type)) }
  def presence
    self if present?
  end
end

class NilClass
  # +nil+ is blank:
  #
  #   nil.blank? # => true
  sig { returns(TrueClass) }
  def blank? = true

  sig { returns(FalseClass) }
  def present? = false # :nodoc:
end

class FalseClass
  # +false+ is blank:
  #
  #   false.blank? # => true
  sig { returns(TrueClass) }
  def blank? = true

  sig { returns(FalseClass) }
  def present? = false # :nodoc:
end

class TrueClass
  # +true+ is not blank:
  #
  #   true.blank? # => false
  sig { returns(FalseClass) }
  def blank? = false

  sig { returns(TrueClass) }
  def present? = true # :nodoc:
end

class Array
  # An array is blank if it's empty:
  #
  #   [].blank?      # => true
  #   [1,2,3].blank? # => false
  #
  # @return [true, false]
  alias blank? empty?

  sig { returns(T::Boolean) }
  def present? = !empty? # :nodoc:
end

class Hash
  # A hash is blank if it's empty:
  #
  #   {}.blank?                # => true
  #   { key: 'value' }.blank?  # => false
  #
  # @return [true, false]
  alias blank? empty?

  sig { returns(T::Boolean) }
  def present? = !empty? # :nodoc:
end

class Symbol
  # A Symbol is blank if it's empty:
  #
  #   :''.blank?     # => true
  #   :symbol.blank? # => false
  alias blank? empty?

  sig { returns(T::Boolean) }
  def present? = !empty? # :nodoc:
end

class String
  BLANK_RE = /\A[[:space:]]*\z/
  # This is a cache that is intentionally mutable
  # rubocop:disable Style/MutableConstant
  ENCODED_BLANKS_ = T.let(Hash.new do |h, enc|
    h[enc] = Regexp.new(BLANK_RE.source.encode(enc), BLANK_RE.options | Regexp::FIXEDENCODING)
  end, T::Hash[Encoding, Regexp])
  # rubocop:enable Style/MutableConstant

  # A string is blank if it's empty or contains whitespaces only:
  #
  #   ''.blank?       # => true
  #   '   '.blank?    # => true
  #   "\t\n\r".blank? # => true
  #   ' blah '.blank? # => false
  #
  # Unicode whitespace is supported:
  #
  #   "\u00a0".blank? # => true
  sig { returns(T::Boolean) }
  def blank?
    # The regexp that matches blank strings is expensive. For the case of empty
    # strings we can speed up this method (~3.5x) with an empty? call. The
    # penalty for the rest of strings is marginal.
    empty? ||
      begin
        BLANK_RE.match?(self)
      rescue Encoding::CompatibilityError
        T.must(ENCODED_BLANKS_[encoding]).match?(self)
      end
  end

  sig { returns(T::Boolean) }
  def present? = !blank? # :nodoc:
end

class Numeric # :nodoc:
  # No number is blank:
  #
  #   1.blank? # => false
  #   0.blank? # => false
  sig { returns(FalseClass) }
  def blank? = false

  sig { returns(TrueClass) }
  def present? = true
end

class Time # :nodoc:
  # No Time is blank:
  #
  #   Time.now.blank? # => false
  sig { returns(FalseClass) }
  def blank? = false

  sig { returns(TrueClass) }
  def present? = true
end
