# typed: strict
# frozen_string_literal: true

# Utility method extensions for String.
class String
  extend T::Sig

  sig { returns(String) }
  def undent
    gsub(/^.{#{(slice(/^ +/) || '').length}}/, "")
  end
end
