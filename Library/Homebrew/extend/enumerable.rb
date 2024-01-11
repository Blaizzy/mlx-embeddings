# typed: strict
# frozen_string_literal: true

module Enumerable
  # Convert an enumerable to a hash, using the block result as the key and the
  # element as the value.
  #
  #   people.index_by(&:login)
  #   # => { "nextangle" => <Person ...>, "chade-" => <Person ...>, ...}
  #
  #   people.index_by { |person| "#{person.first_name} #{person.last_name}" }
  #   # => { "Chade- Fowlersburg-e" => <Person ...>, "David Heinemeier Hansson" => <Person ...>, ...}
  def index_by(&block)
    if block
      result = {}
      each { |elem| result[yield(elem)] = elem }
      result
    else
      T.unsafe(self).to_enum(:index_by) { T.unsafe(self).size if respond_to?(:size) }
    end
  end

  # The negative of the <tt>Enumerable#include?</tt>. Returns +true+ if the
  # collection does not include the object.
  sig { params(object: T.untyped).returns(T::Boolean) }
  def exclude?(object) = !include?(object)

  # Returns a new +Array+ without the blank items.
  # Uses Object#blank? for determining if an item is blank.
  #
  #   [1, "", nil, 2, " ", [], {}, false, true].compact_blank
  #   # =>  [1, 2, true]
  #
  #   Set.new([nil, "", 1, false]).compact_blank
  #   # => [1]
  #
  # When called on a +Hash+, returns a new +Hash+ without the blank values.
  #
  #   { a: "", b: 1, c: nil, d: [], e: false, f: true }.compact_blank
  #   # => { b: 1, f: true }
  sig { returns(T.self_type) }
  def compact_blank = T.unsafe(self).reject(&:blank?)
end

class Hash
  # Hash#reject has its own definition, so this needs one too.
  def compact_blank = reject { T.unsafe(_2).blank? }
end
