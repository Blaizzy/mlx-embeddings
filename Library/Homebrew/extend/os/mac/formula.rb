# typed: true
# frozen_string_literal: true

class Formula
  undef valid_platform?

  sig { returns(T::Boolean) }
  def valid_platform?
    requirements.none?(LinuxRequirement)
  end
end
