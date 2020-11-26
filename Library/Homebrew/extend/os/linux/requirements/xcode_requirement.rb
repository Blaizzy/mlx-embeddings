# typed: strict
# frozen_string_literal: true

require "requirement"

class XcodeRequirement < Requirement
  extend T::Sig

  sig { returns(T::Boolean) }
  def xcode_installed_version
    true
  end
end
