# typed: false
# frozen_string_literal: true

require "delegate"

# A collection of dependencies.
#
# @api private
class Dependencies < SimpleDelegator
  extend T::Sig

  def initialize(*args)
    super(args)
  end

  alias eql? ==

  def optional
    select(&:optional?)
  end

  def recommended
    select(&:recommended?)
  end

  def build
    select(&:build?)
  end

  def required
    select(&:required?)
  end

  def default
    build + required + recommended
  end

  sig { returns(String) }
  def inspect
    "#<#{self.class.name}: #{to_a}>"
  end
end

# A collection of requirements.
#
# @api private
class Requirements < SimpleDelegator
  extend T::Sig

  def initialize(*args)
    super(Set.new(args))
  end

  def <<(other)
    if other.is_a?(Comparable)
      grep(other.class) do |req|
        return self if req > other

        delete(req)
      end
    end
    super
    self
  end

  sig { returns(String) }
  def inspect
    "#<#{self.class.name}: {#{to_a.join(", ")}}>"
  end
end
