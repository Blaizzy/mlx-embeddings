# typed: strict
# frozen_string_literal: true

class KegOnlyReason
  extend T::Sig
  sig { returns(T::Boolean) }
  def applicable?
    true
  end
end
