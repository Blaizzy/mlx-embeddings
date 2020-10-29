# typed: strict
# frozen_string_literal: true

class BottleSpecification
  extend T::Sig
  sig { returns(T::Boolean) }
  def skip_relocation?
    false
  end
end
